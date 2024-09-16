import torch
import torch.nn as nn
from tqdm import notebook as tqdm_notebook
from tqdm import tqdm

from src.dataset_loader import UserItemRatingDataset, UserItemRatingPairwiseDataset


class LinearRegressionPairwiseRanking(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        metadata_feature_size,
        device,
        dropout_prob=0.3,
    ):
        super(LinearRegressionPairwiseRanking, self).__init__()

        self.device = device

        # Embeddings for users and items with specified embedding dimension
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_prob)

        # Global bias term (mean rating)
        self.global_bias = nn.Parameter(torch.tensor([0.0])).to(self.device)

        # Linear layer to learn weights for metadata features
        self.metadata_layer = nn.Linear(metadata_feature_size, 1).to(self.device)

        # Linear layer for combining user and item embeddings into a single score
        self.embedding_combiner = nn.Linear(embedding_dim, 1).to(self.device)

    def forward(self, user, item, item_metadata):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)
        item_metadata = item_metadata.to(self.device)

        # Get user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # Apply dropout to the embeddings to regularize
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        # Combine user and item embeddings using a linear layer
        combined_embedding = self.embedding_combiner(user_emb * item_emb).squeeze()

        # Pass item metadata through a linear layer
        metadata_contribution = self.metadata_layer(item_metadata).squeeze()
        metadata_contribution = self.dropout(metadata_contribution)

        # Sum of combined user-item embedding, metadata contribution, and global bias
        output = combined_embedding + metadata_contribution + self.global_bias

        return output

    def predict(self, users, items, item_metadata):
        """
        Predict interaction score (rating) for given users and items.
        """
        users = torch.as_tensor(users, dtype=torch.long, device=self.device)
        items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        item_metadata = torch.as_tensor(
            item_metadata, dtype=torch.float, device=self.device
        )
        return self.forward(users, items, item_metadata)

    def predict_train_batch(self, batch_input: dict, device: str = "cpu"):
        users = batch_input["user_id"].to(device)
        pos_items = batch_input["pos_item_id"].to(device)
        pos_item_metadata = batch_input["pos_item_metadata"].to(device)
        neg_items = batch_input["neg_item_id"].to(device)
        neg_item_metadata = batch_input["neg_item_metadata"].to(device)

        pos_predictions = self.predict(users, pos_items, pos_item_metadata)
        neg_predictions = self.predict(users, neg_items, neg_item_metadata)

        return pos_predictions, neg_predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingPairwiseDataset

    def recommend(
        self, users, items, items_metadata, k, top_n=10, progress_bar_type="tqdm"
    ):
        """
        Generate top k item recommendations with scores for each user in the provided list.

        Parameters:
        - users: List or array of user IDs for whom to generate recommendations.
        - items: List or array of item IDs to consider for each user.
        - items_metadata: List or array of item metadata for each item.
        - k: Number of top recommendations to return for each user.
        - top_n: Number of top similar users to consider when predicting scores.
                 (Not utilized in this MatrixFactorization implementation.)
        - progress_bar_type: Type of progress bar to use ('tqdm' or 'tqdm_notebook').

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indice": [user1, user1, user1, ..., user2, user2, ...],
                "recommendation": [item1, item2, item3, ..., item1, item2, ...],
                "score": [score1, score2, score3, ..., score1, score2, ...]
            }
        """
        # Select the appropriate progress bar
        if progress_bar_type == "tqdm":
            progress_bar = tqdm
        elif progress_bar_type == "tqdm_notebook":
            progress_bar = tqdm_notebook.tqdm
        else:
            raise ValueError(
                "Invalid progress_bar_type. Choose 'tqdm' or 'tqdm_notebook'."
            )

        # Ensure the model is in evaluation mode
        self.eval()

        # Move items and items_metadata to the correct device
        items_tensor = torch.tensor(items, dtype=torch.long, device=self.device)
        items_metadata_tensor = torch.tensor(
            items_metadata, dtype=torch.float, device=self.device
        )

        recommendations_dict = {
            "user_indice": [],
            "recommendation": [],
            "score": [],
        }

        with torch.no_grad():
            for user in progress_bar(users, desc="Generating Recommendations"):
                # Create a tensor of the current user repeated for all items
                user_tensor = torch.full(
                    (len(items_tensor),),
                    user,
                    dtype=torch.long,
                    device=self.device,
                )

                # Predict scores for all items for the current user
                scores = self.predict(user_tensor, items_tensor, items_metadata_tensor)

                # Get the top k scores and their corresponding item indices
                top_scores, top_indices = torch.topk(scores, k, largest=True)

                # Convert tensors to CPU numpy arrays for processing
                top_scores = top_scores.cpu().numpy()
                top_indices = top_indices.cpu().numpy()

                # Retrieve the top k item IDs
                top_items = [items[i] for i in top_indices]

                # Append the results to the recommendations dictionary
                recommendations_dict["user_indice"].extend([user] * k)
                recommendations_dict["recommendation"].extend(top_items)
                recommendations_dict["score"].extend(top_scores)

        return recommendations_dict


class LinearRegressionRatingPrediction(LinearRegressionPairwiseRanking):
    def predict_train_batch(self, batch_input: dict, device: str = "cpu"):
        users = batch_input["user"]
        items = batch_input["item"]

        users = users.to(device)
        items = items.to(device)

        predictions = self.predict(users, items)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingDataset
