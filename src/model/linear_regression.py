from typing import Any, Dict

import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDFDataset


class LinearRegressionRatingPrediction(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        item_feature_size,
        dropout=0.2,
    ):
        super().__init__()

        # Embeddings for users and items with specified embedding dimension
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Global bias term (mean rating)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        self.item_feature_layer = nn.Linear(item_feature_size, embedding_dim)

        # Fully connected layer to map concatenated embeddings to rating prediction
        self.fc_rating = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            # nn.BatchNorm1d(embedding_dim),
            self.relu,
            # self.dropout,
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, user, item, item_feature):
        user = user
        item = item
        item_feature = item_feature

        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        item_feature_emb = self.item_feature_layer(item_feature).squeeze(1)

        cat_emb = torch.cat([user_emb, item_emb, item_feature_emb], dim=-1)

        x = self.fc_rating(cat_emb)
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
                items_feature_batch = (
                    item_features.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)
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
