from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.dataset_loader import UserItemRatingDFDataset


class LinearRegressionRatingPrediction(nn.Module):
    """
    A PyTorch-based linear regression model for predicting user-item ratings using embeddings and item features.

    This model uses user and item embeddings and additional item-specific features to predict ratings.
    The architecture includes embedding layers for users and items, a fully connected layer, and a
    global bias term.

    Args:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        embedding_dim (int): The dimension of the user and item embeddings.
        item_feature_size (int): The size of the additional item-specific feature vector.
    """

    def __init__(self, num_users, num_items, embedding_dim, item_feature_size):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        input_size = embedding_dim * 2 + item_feature_size
        self.fc_rating = nn.Sequential(
            nn.Linear(input_size, 1),
        )

    def forward(self, user, item, item_feature):
        """
        Perform the forward pass of the model.

        Args:
            user (torch.Tensor): A tensor of user indices.
            item (torch.Tensor): A tensor of item indices.
            item_feature (torch.Tensor): A tensor of item feature vectors.

        Returns:
            torch.Tensor: The predicted rating for the user-item interaction with features.
        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        input_layer = torch.cat([user_emb, item_emb, item_feature], dim=-1)
        x = self.fc_rating(input_layer)
        output = x + self.global_bias

        return output

    def predict(self, user, item, item_feature):
        """
        Predict the rating for a given user, item, and item feature vector.

        Args:
            user (torch.Tensor): A tensor of user indices.
            item (torch.Tensor): A tensor of item indices.
            item_feature (torch.Tensor): A tensor of item feature vectors.

        Returns:
            torch.Tensor: The predicted rating, passed through a sigmoid function.
        """
        output_rating = self.forward(user, item, item_feature)
        return nn.Sigmoid()(output_rating)

    def predict_train_batch(
        self, batch_input: dict, device: torch.device = torch.device("cpu")
    ):
        """
        Predict ratings for a batch of user-item interactions during training.

        Args:
            batch_input (dict): A dictionary containing the batch data, which includes 'user', 'item',
                                and 'item_feature' tensors.
            device (torch.device): The device to run the model on, default is CPU.

        Returns:
            torch.Tensor: The predicted ratings for the batch.
        """
        user = batch_input["user"].to(device)
        item = batch_input["item"].to(device)
        item_feature = batch_input["item_feature"].to(device)

        predictions = self.forward(user, item, item_feature)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        """
        Returns the expected dataset type that this model works with.

        Returns:
            UserItemRatingDFDataset: The dataset class this model is designed to work with.
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
        Generate top-k recommendations for a batch of users, evaluating all available items.

        Args:
            users (torch.Tensor): A tensor containing the user indices.
            items (torch.Tensor): A tensor containing the item indices for all available items.
            item_features (torch.Tensor): A tensor containing the feature vectors for all items.
            k (int): The number of top recommendations to generate per user.
            batch_size (int): The batch size for generating recommendations, default is 128.

        Returns:
            Dict[str, Any]: A dictionary with three keys:
                - "user_indice": List of user indices for which recommendations were generated.
                - "recommendation": List of recommended item indices for each user.
                - "score": List of recommendation scores corresponding to each recommended item.
        """
        self.eval()
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
