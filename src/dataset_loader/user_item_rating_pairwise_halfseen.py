import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class UserItemRatingPairwiseHalfSeenDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        label_col: str,
        item_metadata=None,
        num_negative_samples=5,
    ):
        # Reset index so that the index corresponds to the row number
        self.df = df.reset_index(drop=True)
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.item_metadata = item_metadata
        self.num_negative_samples = num_negative_samples

        # Group data by user_id for easier sampling
        self.user_groups = self.df.groupby(user_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        # Get the data at the index
        row = self.df.iloc[idx]
        user_id = row[self.user_col]
        pos_item = row[self.item_col]
        pos_label = row[self.label_col]

        # Get a group of data for the same user
        user_group = self.user_groups.get_group(user_id)

        # Find all items the user has interacted with
        user_items = set(user_group[self.item_col])

        # Get negative samples for the same user where label < pos_label
        user_neg_samples = user_group[user_group[self.label_col] < pos_label]
        neg_indices = user_neg_samples.index.tolist()

        neg_sample_indices = []
        neg_labels = []

        # 50% of the negative samples to be from seen interactions
        if self.num_negative_samples == 1:
            num_seen_neg_samples_needed = np.random.randint(2)
        else:
            num_seen_neg_samples_needed = self.num_negative_samples // 2

        if num_seen_neg_samples_needed == 0:
            num_neg_samples_needed = self.num_negative_samples
        elif len(neg_indices) >= num_seen_neg_samples_needed:
            sampled_neg_indices = np.random.choice(
                neg_indices, num_seen_neg_samples_needed, replace=False
            ).tolist()
            neg_sample_indices.extend(sampled_neg_indices)
            neg_labels.extend(self.df.loc[sampled_neg_indices, self.label_col].tolist())
            num_neg_samples_needed = self.num_negative_samples - len(
                sampled_neg_indices
            )
        else:
            # Use all available negative samples from user interactions
            neg_sample_indices.extend(neg_indices)
            neg_labels.extend(self.df.loc[neg_indices, self.label_col].tolist())
            num_neg_samples_needed = self.num_negative_samples - len(neg_indices)

        # Sample additional negative items from unseen items
        unseen_indices = self.df[
            ~self.df[self.item_col].isin(user_items)
        ].index.tolist()
        additional_neg_indices = np.random.choice(
            unseen_indices, num_neg_samples_needed, replace=False
        ).tolist()
        neg_sample_indices.extend(additional_neg_indices)
        neg_labels.extend([0.0] * len(additional_neg_indices))

        # Retrieve negative items
        neg_items = self.df.loc[neg_sample_indices, self.item_col].values

        # Compute labels for MarginRankingLoss
        # labels = np.array([1.0 if pos_label > nl else -1.0 for nl in neg_labels])
        # Below is a version where we weight the label by the actual differences in ratings
        labels = np.array([pos_label - nl for nl in neg_labels])

        # Retrieve item metadata
        pos_item_metadata = []
        neg_item_metadata = []
        if self.item_metadata is not None:
            pos_item_metadata = self.item_metadata[idx]
            neg_item_metadata = [self.item_metadata[i] for i in neg_sample_indices]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item, dtype=torch.long),
            "pos_item_metadata": torch.as_tensor(pos_item_metadata),
            "neg_item_ids": torch.tensor(neg_items, dtype=torch.long),
            "neg_item_metadata": torch.as_tensor(neg_item_metadata),
            "labels": torch.as_tensor(labels, dtype=torch.float32),
        }

    @classmethod
    def get_default_loss_fn(cls):
        return nn.MarginRankingLoss(margin=1.0)

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        user_ids = batch_input["user_id"].to(device)
        pos_item_ids = batch_input["pos_item_id"].to(device)
        neg_item_ids = batch_input["neg_item_ids"].to(device)
        labels = batch_input["labels"].to(device)

        # Ensure that neg_item_ids and labels are 2D (batch_size x num_negative_samples)
        if neg_item_ids.dim() == 1:
            neg_item_ids = neg_item_ids.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        pos_predictions = model.predict(user_ids, pos_item_ids)  # shape [batch_size]

        # Expand pos_predictions to match the shape of neg_predictions
        num_neg_samples = neg_item_ids.shape[1]
        pos_predictions_expanded = (
            pos_predictions.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
        )

        # Flatten neg_item_ids and user_ids for batch prediction
        user_ids_expanded = (
            user_ids.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
        )
        neg_item_ids_flat = neg_item_ids.reshape(-1)

        neg_predictions_flat = model.predict(user_ids_expanded, neg_item_ids_flat)

        labels_flat = labels.reshape(-1)

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()

        loss = loss_fn(pos_predictions_expanded, neg_predictions_flat, labels_flat)

        return loss
