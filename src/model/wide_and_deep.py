from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDataset, UserItemRatingPairwiseFullDataset


class WideAndDeepRatingPrediction(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_units: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Deep part (generalization) - embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Fully connected layers for the deep part
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.fc3 = nn.Linear(hidden_units // 2 + 2 * embedding_dim, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        user_item_emb = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through the deep layers
        x = self.fc1(user_item_emb)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        wide_deep_output = torch.cat([user_item_emb, x], dim=-1)
        output = self.fc3(wide_deep_output)
        output = self.sigmoid(output)

        return output

    # class WideAndDeepRatingPrediction(nn.Module):
    #     def __init__(
    #         self,
    #         num_users: int,
    #         num_items: int,
    #         embedding_dim: int,
    #         hidden_units: int,
    #         dropout: float = 0.2,
    #         device: torch.device = torch.device("cpu"),
    #     ):
    #         super(WideAndDeepRatingPrediction, self).__init__()

    #         self.device = device

    #         # Deep part (generalization) - embeddings for users and items
    #         self.user_embedding = nn.Embedding(num_users, embedding_dim)
    #         self.item_embedding = nn.Embedding(num_items, embedding_dim)

    #         # Fully connected layers for the deep part
    #         self.fc1 = nn.Linear(2 * embedding_dim, hidden_units)
    #         self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
    #         self.fc3 = nn.Linear(hidden_units // 2 + embedding_dim * 2, 1)

    #         # Activation and dropout
    #         self.relu = nn.ReLU()
    #         self.dropout = nn.Dropout(dropout)
    #         self.sigmoid = nn.Sigmoid()

    #         # Move all layers to the specified device
    #         self.to(self.device)

    #     def forward(self, user, item) -> torch.Tensor:
    #         """
    #         Forward pass for WideAndDeep model.

    #         Args:
    #             user (torch.Tensor): Tensor of user indices.
    #             item (torch.Tensor): Tensor of item indices.

    #         Returns:
    #             torch.Tensor: Predicted scores.
    #         """
    #         # Ensure input tensors are on the correct device
    #         user = torch.as_tensor(user).to(self.device)
    #         item = torch.as_tensor(item).to(self.device)

    #         user_emb = self.user_embedding(user)
    #         item_emb = self.item_embedding(item)
    #         user_item_emb = torch.cat([user_emb, item_emb], dim=-1)

    #         # Pass through the deep layers
    #         x = self.fc1(user_item_emb)
    #         x = self.relu(x)
    #         x = self.dropout(x)
    #         x = self.fc2(x)
    #         x = self.relu(x)
    #         x = self.dropout(x)
    #         wide_deep_output = torch.cat([user_item_emb, x], dim=-1)
    #         output = self.fc3(wide_deep_output)
    #         output = self.sigmoid(output)

    #         return output

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict interaction score for given users and items.

        Args:
            users (torch.Tensor): Tensor of user indices.
            items (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Predicted scores.
        """
        return self.forward(users, items)

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
        return self.predict(users, items)

    @classmethod
    def get_expected_dataset_type(cls):
        """
        Returns the expected dataset type for this model.

        Returns:
            Dataset class
        """
        return UserItemRatingDataset

    def recommend(
        self,
        users: torch.Tensor,
        k: int,
        batch_size: int = 128,
        progress_bar_type: str = "tqdm",
    ) -> Dict[str, Any]:
        if progress_bar_type == "tqdm_notebook":
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm

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


#     def recommend(
#         self,
#         users: torch.Tensor,
#         k: int,
#         batch_size: int = 128,
#         progress_bar_type: str = "tqdm",
#     ) -> Dict[str, Any]:
#         """
#         Generate top k item recommendations with scores for each user in the provided list.
#         """
#         if progress_bar_type == "tqdm_notebook":
#             from tqdm.notebook import tqdm
#         else:
#             from tqdm import tqdm

#         self.eval()  # Set model to evaluation mode
#         all_items = torch.arange(self.item_embedding.num_embeddings).to(self.device)

#         user_indices = []
#         recommendations = []
#         scores = []

#         with torch.no_grad():
#             total_users = len(users)
#             for i in tqdm(
#                 range(0, total_users, batch_size), desc="Generating recommendations"
#             ):
#                 user_batch = users[i : i + batch_size]
#                 user_batch = torch.tensor(user_batch, dtype=torch.long).to(self.device)

#                 # Expand user_batch to match all items
#                 user_batch_expanded = (
#                     user_batch.unsqueeze(1).repeat(1, len(all_items)).view(-1)
#                 )
#                 items_batch = all_items.repeat(len(user_batch))

#                 # Predict scores for the batch
#                 batch_scores = self.predict(user_batch_expanded, items_batch).view(
#                     len(user_batch), -1
#                 )

#                 # Get top k items for each user in the batch
#                 topk_scores, topk_indices = torch.topk(batch_scores, k, dim=1)
#                 topk_items = all_items[topk_indices]

#                 # Collect recommendations
#                 user_indices.extend(np.repeat(user_batch.cpu().numpy(), k).tolist())
#                 recommendations.extend(topk_items.cpu().numpy().flatten().tolist())
#                 scores.extend(topk_scores.cpu().numpy().flatten().tolist())

#         recommendations_dict = {
#             "user_indice": user_indices,
#             "recommendation": recommendations,
#             "score": scores,
#         }

#         return recommendations_dict


# class WideAndDeepPairwiseRanking(WideAndDeepRatingPrediction):
#     def predict_train_batch(
#         self, batch_input: Dict[str, Any]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns positive and negative predictions for a batch during training.

#         Args:
#             batch_input (Dict[str, Any]): Dictionary containing:
#                 - 'user_indice': Tensor of user indices.
#                 - 'pos_item_id': Tensor of positive item IDs.
#                 - 'neg_item_ids': Tensor of negative item IDs (can be multiple per positive sample).

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]:
#                 - pos_predictions: Predictions for positive items.
#                 - neg_predictions: Predictions for negative items.
#         """
#         # Extract and move data to the correct device
#         user_ids = batch_input["user_indice"].to(self.device)
#         pos_item = batch_input["pos_item_id"].to(self.device)
#         neg_items = batch_input["neg_item_ids"].to(
#             self.device
#         )  # Shape: [batch_size, num_negatives]

#         # Ensure that neg_items is 2D (batch_size x num_negative_samples)
#         if neg_items.dim() == 1:
#             neg_items = neg_items.unsqueeze(1)  # Convert to shape [batch_size, 1]

#         batch_size = pos_item.size(0)
#         num_neg_samples = neg_items.size(1)

#         # Positive predictions
#         pos_predictions = self.predict(
#             users=user_ids, items=pos_item
#         )  # Shape: [batch_size]

#         # Prepare for negative predictions
#         # Expand user_ids to [batch_size * num_neg_samples]
#         user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_neg_samples)
#         user_ids_flat = user_ids_expanded.reshape(
#             -1
#         )  # Shape: [batch_size * num_neg_samples]

#         # Flatten negative items to [batch_size * num_neg_samples]
#         neg_items_flat = neg_items.reshape(-1)  # Shape: [batch_size * num_neg_samples]

#         # Negative predictions
#         neg_predictions_flat = self.predict(
#             users=user_ids_flat, items=neg_items_flat
#         )  # Shape: [batch_size * num_neg_samples]

#         # Reshape neg_predictions back to [batch_size, num_neg_samples]
#         neg_predictions = neg_predictions_flat.view(
#             batch_size, num_neg_samples
#         )  # Shape: [batch_size, num_neg_samples]

#         return pos_predictions, neg_predictions

#     @classmethod
#     def get_expected_dataset_type(cls):
#         """
#         Returns the expected dataset type for pairwise ranking.

#         Returns:
#             Dataset class
#         """
#         return UserItemRatingPairwiseFullDataset
