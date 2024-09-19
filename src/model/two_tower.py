from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from src.dataset_loader import (
    UserItemRatingDataset,
    UserItemRatingPairwiseDataset,
    UserItemRatingPairwiseFullDataset,
)


class TwoTowerPairwiseRanking(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        hidden_units,
        dropout: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        super(TwoTowerPairwiseRanking, self).__init__()

        self.device = device

        # User Tower: embedding and MLP for user features
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.user_fc1 = nn.Linear(embedding_dim, hidden_units).to(self.device)
        self.user_fc2 = nn.Linear(hidden_units, hidden_units // 2).to(self.device)

        # Item Tower: embedding and MLP for item features
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)
        self.item_fc1 = nn.Linear(embedding_dim, hidden_units).to(self.device)
        self.item_fc2 = nn.Linear(hidden_units, hidden_units // 2).to(self.device)

        # Activation and dropout
        self.relu = nn.ReLU().to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, user, item):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)

        # User Tower
        user_emb = self.user_embedding(user)
        user_x = self.user_fc1(user_emb)
        user_x = self.relu(user_x)
        user_x = self.dropout(user_x)
        user_x = self.user_fc2(user_x)
        user_x = self.relu(user_x)

        # Item Tower
        item_emb = self.item_embedding(item)
        item_x = self.item_fc1(item_emb)
        item_x = self.relu(item_x)
        item_x = self.dropout(item_x)
        item_x = self.item_fc2(item_x)
        item_x = self.relu(item_x)

        # Final output is the dot product of the two towers' outputs
        output = torch.sum(user_x * item_x, dim=1).unsqueeze(1)

        return output

    def predict(self, users, items):
        """
        Predict interaction score for given users and items.
        """
        users = torch.as_tensor(users)
        items = torch.as_tensor(items)
        return self.forward(users, items)

    def predict_train_batch(
        self, batch_input: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns positive and negative predictions for a batch during training.

        Args:
            batch_input (Dict[str, Any]): Dictionary containing:
                - 'user_indice': Tensor of user indices.
                - 'pos_item_id': Tensor of positive item IDs.
                - 'neg_item_ids': Tensor of negative item IDs (can be multiple per positive sample).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pos_predictions: Predictions for positive items.
                - neg_predictions: Predictions for negative items.
        """
        # Extract and move data to the correct device
        user_ids = batch_input["user_indice"].to(self.device)
        pos_item = batch_input["pos_item_id"].to(self.device)
        neg_items = batch_input["neg_item_ids"].to(
            self.device
        )  # Shape: [batch_size, num_negatives]

        # Ensure that neg_items is 2D (batch_size x num_negative_samples)
        if neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)  # Convert to shape [batch_size, 1]

        batch_size = pos_item.size(0)
        num_neg_samples = neg_items.size(1)

        # Positive predictions
        pos_predictions = self.predict(
            users=user_ids, items=pos_item
        )  # Shape: [batch_size]

        # Prepare for negative predictions
        # Expand user_ids to [batch_size * num_neg_samples]
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_neg_samples)
        user_ids_flat = user_ids_expanded.reshape(
            -1
        )  # Shape: [batch_size * num_neg_samples]

        # Flatten negative items to [batch_size * num_neg_samples]
        neg_items_flat = neg_items.reshape(-1)  # Shape: [batch_size * num_neg_samples]

        # Negative predictions
        neg_predictions_flat = self.predict(
            users=user_ids_flat, items=neg_items_flat
        )  # Shape: [batch_size * num_neg_samples]

        # Reshape neg_predictions back to [batch_size, num_neg_samples]
        neg_predictions = neg_predictions_flat.view(
            batch_size, num_neg_samples
        )  # Shape: [batch_size, num_neg_samples]

        return pos_predictions, neg_predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingPairwiseFullDataset

    def recommend(self, users, k, batch_size=128, progress_bar_type="tqdm"):
        """
        Generate top k item recommendations with scores for each user in the provided list.

        Parameters:
        - users: List or array of user IDs for whom to generate recommendations.
        - k: Number of top recommendations to return for each user.
        - progress_bar_type: Type of progress bar to use ('tqdm' or 'tqdm_notebook').

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indice": [user1, user1, user1, ..., user2, user2, ...],
                "recommendation": [item1, item2, item3, ..., item1, item2, ...],
                "score": [score1, score2, score3, ..., score1, score2, ...]
            }
        """
        # Select the appropriate tqdm function based on progress_bar_type
        if progress_bar_type == "tqdm":
            progress_bar_func = tqdm
        elif progress_bar_type == "tqdm_notebook":
            progress_bar_func = tqdm_notebook
        else:
            raise ValueError(
                "progress_bar_type must be either 'tqdm' or 'tqdm_notebook'"
            )

        # Set the model to evaluation mode
        self.eval()

        # Initialize the recommendations dictionary
        recommendations_dict = {"user_indice": [], "recommendation": [], "score": []}

        with torch.no_grad():
            # Precompute item representations
            num_items = self.item_embedding.num_embeddings
            all_items = torch.arange(num_items, dtype=torch.long).to(self.device)
            item_emb = self.item_embedding(all_items)
            item_x = self.item_fc1(item_emb)
            item_x = self.relu(item_x)
            item_x = self.dropout(item_x)
            item_x = self.item_fc2(item_x)
            item_x = self.relu(item_x)  # Shape: (num_items, hidden_units // 2)

            # Process users in batches with a progress bar
            for i in progress_bar_func(
                range(0, len(users), batch_size), desc="Generating recommendations"
            ):
                batch_users = users[i : i + batch_size]
                users_tensor = torch.tensor(batch_users, dtype=torch.long).to(
                    self.device
                )

                # User Tower
                user_emb = self.user_embedding(users_tensor)
                user_x = self.user_fc1(user_emb)
                user_x = self.relu(user_x)
                user_x = self.dropout(user_x)
                user_x = self.user_fc2(user_x)
                user_x = self.relu(user_x)  # Shape: (batch_size, hidden_units // 2)

                # Compute scores by performing a dot product between user and item representations
                # Resulting shape: (batch_size, num_items)
                scores = torch.matmul(user_x, item_x.t())

                # Get top k items for each user in the batch
                topk_scores, topk_indices = torch.topk(scores, k, dim=1)

                # Move tensors to CPU and convert to NumPy for further processing
                topk_scores = topk_scores.cpu().numpy()
                topk_indices = topk_indices.cpu().numpy()

                # Flatten the results into the recommendations dictionary
                for user_id, item_ids, scores in zip(
                    batch_users, topk_indices, topk_scores
                ):
                    recommendations_dict["user_indice"].extend([user_id] * k)
                    recommendations_dict["recommendation"].extend(item_ids.tolist())
                    recommendations_dict["score"].extend(scores.tolist())

        return recommendations_dict


class TwoTowerRatingPrediction(TwoTowerPairwiseRanking):
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
