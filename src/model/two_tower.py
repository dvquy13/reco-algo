from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.dataset_loader import UserItemRatingDFDataset


class TwoTowerRatingPrediction(nn.Module):
    """
    A two-tower neural network for rating prediction in recommender systems.

    This model consists of two separate towers for users and items. Each tower
    contains an embedding layer followed by a fully connected (FC) layer with
    ReLU activation and dropout. The final predicted score is computed as the
    dot product between the user and item towers' outputs.

    Attributes:
        user_embedding (nn.Embedding): Embedding layer for user indices.
        user_fc1 (nn.Linear): First fully connected layer for the user tower.
        item_embedding (nn.Embedding): Embedding layer for item indices.
        item_fc1 (nn.Linear): First fully connected layer for the item tower.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self, num_users, num_items, embedding_dim, hidden_units, dropout: float = 0.2
    ):
        """
        Initializes the TwoTowerRatingPrediction model.

        Args:
            num_users (int): The number of unique users.
            num_items (int): The number of unique items.
            embedding_dim (int): Dimensionality of user and item embeddings.
            hidden_units (int): Number of hidden units in the fully connected layers.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
        """
        super().__init__()

        # User Tower: embedding and MLP for user features
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_fc1 = nn.Linear(embedding_dim, hidden_units)

        # Item Tower: embedding and MLP for item features
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_fc1 = nn.Linear(embedding_dim, hidden_units)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, user, item):
        """
        Forward pass of the two-tower model.

        Args:
            user (torch.Tensor): Tensor of user indices.
            item (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Predicted interaction scores for the user-item pairs.
        """
        # User Tower
        user_emb = self.user_embedding(user)
        user_x = self.user_fc1(user_emb)
        user_x = self.relu(user_x)
        user_x = self.dropout(user_x)

        # Item Tower
        item_emb = self.item_embedding(item)
        item_x = self.item_fc1(item_emb)
        item_x = self.relu(item_x)
        item_x = self.dropout(item_x)

        # Compute dot product between user and item tower outputs
        output = torch.sum(user_x * item_x, dim=1)

        return output

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict interaction score for given users and items.

        Args:
            users (torch.Tensor): Tensor of user indices.
            items (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Predicted scores.
        """
        output_ratings = self.forward(users, items)
        return nn.Sigmoid()(output_ratings)

    def predict_train_batch(
        self, batch_input: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Predict scores for a training batch.

        Args:
            batch_input (Dict[str, Any]): Dictionary containing 'user' and 'item' tensors.
            device (torch.device, optional): Device to perform computations on. Defaults to CPU.

        Returns:
            torch.Tensor: Predicted scores.
        """
        users = batch_input["user"].to(device)
        items = batch_input["item"].to(device)
        return self.forward(users, items)

    @classmethod
    def get_expected_dataset_type(cls):
        """
        Returns the expected dataset type for this model.

        Returns:
            UserItemRatingDFDataset: The dataset type used by this model.
        """
        return UserItemRatingDFDataset

    def recommend(
        self, users: torch.Tensor, k: int, batch_size: int = 128
    ) -> Dict[str, Any]:
        """
        Generate top-k item recommendations for each user.

        Args:
            users (torch.Tensor): Tensor of user indices.
            k (int): Number of top items to recommend for each user.
            batch_size (int, optional): Batch size for processing users. Defaults to 128.

        Returns:
            Dict[str, Any]: Dictionary containing recommended items and scores:
                'user_indice': List of user indices.
                'recommendation': List of recommended item indices.
                'score': List of predicted interaction scores.
        """
        self.eval()
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
