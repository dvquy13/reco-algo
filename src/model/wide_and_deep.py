from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.dataset_loader import UserItemRatingDFDataset


class WideAndDeepRatingPrediction(nn.Module):
    """
    Wide and Deep model for rating prediction using user embeddings, item embeddings,
    and additional item features. The model combines a wide component (concatenation
    of embeddings and features) and a deep component (a multi-layer neural network).

    Args:
        num_users (int): Number of users.
        num_items (int): Number of items.
        embedding_dim (int): Dimensionality of the user and item embeddings.
        hidden_units (int): Number of hidden units in the deep layers.
        item_feature_size (int): Size of the additional item feature vector.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_units: int,
        item_feature_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_size = 2 * embedding_dim + item_feature_size
        self.fc1 = nn.Linear(input_size, input_size // 2)

        self.fc2 = nn.Linear(input_size // 2, hidden_units)

        self.fc3 = nn.Linear(hidden_units + input_size, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, user: torch.Tensor, item: torch.Tensor, item_feature: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for rating prediction.

        Args:
            user (torch.Tensor): Tensor of user indices.
            item (torch.Tensor): Tensor of item indices.
            item_feature (torch.Tensor): Tensor of additional item feature vectors.

        Returns:
            torch.Tensor: Predicted rating.
        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        wide_layer = torch.cat([user_emb, item_emb, item_feature], dim=-1)

        # Pass through the deep layers
        x = self.fc1(wide_layer)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        wide_deep_output = torch.cat([wide_layer, x], dim=-1)
        output = self.fc3(wide_deep_output)

        return output

    def predict(
        self, user: torch.Tensor, item: torch.Tensor, item_feature: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict interaction score for given users and items.

        Args:
            user (torch.Tensor): Tensor of user indices.
            item (torch.Tensor): Tensor of item indices.
            item_feature (torch.Tensor): Tensor of additional item feature vectors.

        Returns:
            torch.Tensor: Predicted scores as a sigmoid-activated output.
        """
        output_rating = self.forward(user, item, item_feature)
        return nn.Sigmoid()(output_rating)

    def predict_train_batch(
        self, batch_input: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Predict scores for a training batch.

        Args:
            batch_input (Dict[str, Any]): Dictionary containing 'user', 'item', and
                'item_feature' tensors.
            device (torch.device, optional): Device to perform computations on. Defaults to CPU.

        Returns:
            torch.Tensor: Predicted scores.
        """
        user = batch_input["user"].to(device)
        item = batch_input["item"].to(device)
        item_feature = batch_input["item_feature"].to(device)
        return self.forward(user, item, item_feature)

    @classmethod
    def get_expected_dataset_type(cls):
        """
        Returns the expected dataset type for this model.

        Returns:
            type: The class of the expected dataset.
        """
        return UserItemRatingDFDataset

    def recommend(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_features: torch.Tensor,
        k: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """
        Generate top-k recommendations for each user.

        Args:
            users (torch.Tensor): Tensor of user indices.
            items (torch.Tensor): Tensor of item indices to recommend from.
            item_features (torch.Tensor): Tensor of item feature vectors.
            k (int): Number of recommendations per user.
            batch_size (int, optional): Batch size for processing. Defaults to 128.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'user_indice': List of user indices corresponding to recommendations.
                - 'recommendation': List of recommended item indices for each user.
                - 'score': List of scores associated with each recommendation.
        """
        self.eval()  # Set model to evaluation mode
        all_items = items

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
                items_feature_batch = item_features.unsqueeze(0).repeat(
                    len(user_batch), 1, 1
                )
                items_feature_batch = items_feature_batch.view(
                    -1, items_feature_batch.size(-1)
                )

                # Predict scores for the batch
                batch_scores = self.predict(
                    user_batch_expanded, items_batch, items_feature_batch
                ).view(len(user_batch), -1)

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
