from typing import Any, Dict

import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDFDataset


class MatrixFactorizationRatingPrediction(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        # Get user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # Dot product of user and item embeddings to get the predicted rating
        output = torch.sum(user_emb * item_emb, dim=-1)

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
        return UserItemRatingDFDataset

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
