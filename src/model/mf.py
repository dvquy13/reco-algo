import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

from src.dataset_loader import UserItemRatingDataset


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, device):
        super(MatrixFactorization, self).__init__()

        self.device = device

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)

    def forward(self, user, item):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)

        # Get user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # Dot product of user and item embeddings to get the predicted rating
        output = torch.sum(user_emb * item_emb, dim=-1)

        return output

    def predict(self, users, items):
        """
        Predict interaction score (rating) for given users and items.
        """
        users = torch.as_tensor(users)
        items = torch.as_tensor(items)
        return self.forward(users, items)

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

    def recommend(self, users, k, top_n=10, progress_bar_type="tqdm"):
        """
        Generate top k item recommendations with scores for each user in the provided list.

        Parameters:
        - users: List or array of user IDs for whom to generate recommendations.
        - k: Number of top recommendations to return for each user.
        - top_n: Number of top similar users to consider when predicting scores.
                 (Not utilized in this MatrixFactorization implementation.)
        - progress_bar_type: Type of progress bar to use ('tqdm' or 'tqdm_notebook').

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indices": [user1, user1, user1, ..., user2, user2, ...],
                "recommendations": [item1, item2, item3, ..., item1, item2, ...],
                "scores": [score1, score2, score3, ..., score1, score2, ...]
            }
        """
        # Select the appropriate tqdm function based on progress_bar_type
        if progress_bar_type == "tqdm":
            progress_bar = tqdm
        elif progress_bar_type == "tqdm_notebook":
            progress_bar = tqdm_notebook
        else:
            raise ValueError(
                "progress_bar_type must be either 'tqdm' or 'tqdm_notebook'"
            )

        user_indices = []
        all_recommendations = []
        all_scores = []

        # Ensure the model is in evaluation mode
        self.eval()

        with torch.no_grad():
            for user in progress_bar(users, desc="Generating Recommendations"):
                user_tensor = torch.tensor([user], dtype=torch.long).to(self.device)
                user_emb = self.user_embedding(user_tensor)  # Shape: [1, embedding_dim]

                # Compute scores for all items: [1, embedding_dim] x [num_items, embedding_dim]^T = [1, num_items]
                scores = torch.matmul(user_emb, self.item_embedding.weight.T).squeeze(
                    0
                )  # Shape: [num_items]

                # Convert scores to CPU and numpy for processing
                scores = scores.cpu().numpy()

                # Get the indices of the top k scores
                top_k_indices = np.argpartition(-scores, k)[:k]
                top_k_scores = scores[top_k_indices]

                # Sort the top k items by score in descending order
                sorted_top_k_indices = top_k_indices[np.argsort(-top_k_scores)]
                sorted_top_k_scores = scores[sorted_top_k_indices]

                # Append to the results
                user_indices.extend([user] * k)
                all_recommendations.extend(sorted_top_k_indices.tolist())
                all_scores.extend(sorted_top_k_scores.tolist())

        recommendations_dict = {
            "user_indice": user_indices,
            "recommendation": all_recommendations,
            "score": all_scores,
        }

        return recommendations_dict
