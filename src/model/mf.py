from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.dataset_loader import UserItemRatingDFDataset


class MatrixFactorizationRatingPrediction(nn.Module):
    """
    A matrix factorization model for rating prediction using embeddings for users and items.
    This model computes the dot product between user and item embeddings to predict interaction scores.

    Attributes:
        user_embedding (torch.nn.Embedding): Embedding layer for users.
        item_embedding (torch.nn.Embedding): Embedding layer for items.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        """
        Initializes the MatrixFactorizationRatingPrediction model.

        Args:
            num_users (int): The number of unique users.
            num_items (int): The number of unique items.
            embedding_dim (int): The dimensionality of the embedding space.
        """
        super().__init__()

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model to predict the interaction score between user and item embeddings.

        Args:
            user (torch.Tensor): Tensor containing user indices.
            item (torch.Tensor): Tensor containing item indices.

        Returns:
            torch.Tensor: The predicted interaction scores.
        """
        # Get user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # Dot product of user and item embeddings to get the predicted rating
        output = torch.sum(user_emb * item_emb, dim=-1)

        return output

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict interaction scores for the given users and items using the forward method.

        Args:
            users (torch.Tensor): Tensor containing user indices.
            items (torch.Tensor): Tensor containing item indices.

        Returns:
            torch.Tensor: Predicted interaction scores.
        """
        output_ratings = self.forward(users, items)
        return nn.Sigmoid()(output_ratings)

    def predict_train_batch(
        self, batch_input: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Predict scores for a batch of training data.

        Args:
            batch_input (Dict[str, Any]): A dictionary containing tensors with 'user' and 'item' indices.
            device (torch.device, optional): The device on which the model will run (CPU by default).

        Returns:
            torch.Tensor: The predicted scores for the batch.
        """
        users = batch_input["user"].to(device)
        items = batch_input["item"].to(device)
        return self.forward(users, items)

    @classmethod
    def get_expected_dataset_type(cls):
        """
        Returns the expected dataset type for training this model.

        Returns:
            UserItemRatingDFDataset: The class for the expected dataset.
        """
        return UserItemRatingDFDataset

    def recommend(
        self, users: torch.Tensor, k: int, batch_size: int = 128
    ) -> Dict[str, Any]:
        """
        Generate top-k recommendations for the given users.

        This method predicts scores for all items and ranks the top-k items for each user.

        Args:
            users (torch.Tensor): Tensor containing user indices.
            k (int): The number of items to recommend for each user.
            batch_size (int, optional): The batch size for inference. Defaults to 128.

        Returns:
            Dict[str, Any]: A dictionary containing user indices, recommendations, and scores.
                - "user_indice" (List[int]): List of user indices for the recommendations.
                - "recommendation" (List[int]): List of recommended item indices.
                - "score" (List[float]): List of scores for the recommendations.
        """
        self.eval()  # Set model to evaluation mode
        all_items = torch.arange(
            self.item_embedding.num_embeddings, device=users.device
        )

        user_indices = []
        recommendations = []
        scores = []

        with torch.no_grad():
            total_users = users.size(0)
            for i in tqdm(
                range(0, total_users, batch_size), desc="Generating recommendations"
            ):
                user_batch = users[i : i + batch_size]

                # Expand user_batch to match all items
                user_batch_expanded = (
                    user_batch.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)
                )
                items_batch = (
                    all_items.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)
                )

                # Predict scores for the batch
                batch_scores = self.predict(user_batch_expanded, items_batch).view(
                    len(user_batch), -1
                )

                # Get top k items for each user in the batch
                topk_scores, topk_indices = torch.topk(batch_scores, k, dim=1)
                topk_items = all_items[topk_indices]

                # Collect recommendations
                user_indices.extend(user_batch.repeat_interleave(k).cpu().tolist())
                recommendations.extend(topk_items.cpu().flatten().tolist())
                scores.extend(topk_scores.cpu().flatten().tolist())

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }
