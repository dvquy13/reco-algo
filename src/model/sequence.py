from typing import Any, Dict

import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDFDataset


class SequenceRatingPrediction(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        dropout=0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users

        # Item embedding (Add 1 to num_items for the unknown item (-1 padding))
        self.item_embedding = nn.Embedding(
            num_items + 1,  # One additional index for unknown/padding item
            embedding_dim,
            padding_idx=num_items,  # The additional index for the unknown item
        )

        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected layer to map concatenated embeddings to rating prediction
        self.fc_rating = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            # nn.BatchNorm1d(embedding_dim),
            self.relu,
            # self.dropout,
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, user_ids, input_seq, target_item):
        # Replace -1 in input_seq and target_item with num_items (padding_idx)
        padding_idx_tensor = torch.tensor(self.item_embedding.padding_idx)
        input_seq = torch.where(input_seq == -1, padding_idx_tensor, input_seq)
        target_item = torch.where(target_item == -1, padding_idx_tensor, target_item)

        # Embed input sequence
        embedded_seq = self.item_embedding(
            input_seq
        )  # Shape: [batch_size, seq_len, embedding_dim]
        # Mean pooling: take the mean over the sequence dimension (dim=1)
        pooled_output = embedded_seq.mean(dim=1)  # Shape: [batch_size, embedding_dim]

        # Embed the target item
        embedded_target = self.item_embedding(
            target_item
        )  # Shape: [batch_size, embedding_dim]

        # Embed the user IDs
        user_embeddings = self.user_embedding(
            user_ids
        )  # Shape: [batch_size, embedding_dim]

        # Concatenate the pooled sequence output with the target item and user embeddings
        combined_embedding = torch.cat(
            (pooled_output, embedded_target, user_embeddings), dim=1
        )  # Shape: [batch_size, embedding_dim*3]

        # Project combined embedding to rating prediction
        output_ratings = self.fc_rating(combined_embedding)  # Shape: [batch_size, 1]

        return output_ratings  # Shape: [batch_size]

    def predict(self, user, item_sequence, target_item):
        output_ratings = self.forward(user, item_sequence, target_item)
        return nn.Sigmoid()(output_ratings)

    def predict_train_batch(
        self, batch_input: dict, device: torch.device = torch.device("cpu")
    ):
        user = batch_input["user"].to(device)
        item_sequence = batch_input["item_sequence"].to(device)
        target_item = batch_input["item"].to(device)

        predictions = self.forward(user, item_sequence, target_item)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingDFDataset

    def recommend(
        self,
        users: torch.Tensor,
        item_sequences: torch.Tensor,
        k: int,
        batch_size: int = 128,
        progress_bar_type: str = "tqdm",
    ) -> Dict[str, Any]:
        if progress_bar_type == "tqdm_notebook":
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm

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
                item_sequence_batch = item_sequences[i : i + batch_size]

                # Expand user_batch to match all items
                user_batch_expanded = (
                    user_batch.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)
                )
                items_batch = (
                    all_items.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)
                )
                item_sequences_batch = item_sequence_batch.unsqueeze(1).repeat(
                    1, len(all_items), 1
                )
                item_sequences_batch = item_sequences_batch.view(
                    -1, item_sequence_batch.size(-1)
                )

                # Predict scores for the batch
                batch_scores = self.predict(
                    user_batch_expanded, item_sequences_batch, items_batch
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
