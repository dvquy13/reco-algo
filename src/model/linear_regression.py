from typing import Any, Dict

import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDFDataset


class LinearRegressionRatingPrediction(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, item_feature_size):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        input_size = embedding_dim * 2 + item_feature_size
        self.fc_rating = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Linear(input_size // 2, 1),
        )

    def forward(self, user, item, item_feature):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        input_layer = torch.cat([user_emb, item_emb, item_feature], dim=-1)
        x = self.fc_rating(input_layer)
        output = x + self.global_bias

        return output

    def predict(self, user, item, item_feature):
        output_rating = self.forward(user, item, item_feature)
        return nn.Sigmoid()(output_rating)

    def predict_train_batch(
        self, batch_input: dict, device: torch.device = torch.device("cpu")
    ):
        user = batch_input["user"].to(device)
        item = batch_input["item"].to(device)
        item_feature = batch_input["item_feature"].to(device)

        predictions = self.forward(user, item, item_feature)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingDFDataset

    def recommend(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_features: torch.Tensor,
        k: int,
        batch_size: int = 128,
        progress_bar_type: str = "tqdm",
    ) -> Dict[str, Any]:
        if progress_bar_type == "tqdm_notebook":
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm

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
