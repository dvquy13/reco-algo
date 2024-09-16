import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class UserItemRatingPairwiseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        label_col: str,
        item_metadata=None,
    ):
        # Reset index so that the index corresponds to the row number
        self.df = df.reset_index(drop=True)
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.item_metadata = item_metadata

        # Group data by user_id for easier sampling
        self.user_groups = self.df.groupby(user_col)

        # Set of all items to sample from for unseen items
        self.all_items = set(self.df[item_col].unique())

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

        # Sample a negative item for the same user
        user_neg_samples = user_group[user_group[self.label_col] < pos_label]
        if user_neg_samples.shape[0] > 0:
            neg_sample = user_neg_samples.sample(1)
            neg_label = neg_sample[self.label_col].values[0]
        else:
            unseen_rows = self.df[~self.df[self.item_col].isin(user_items)]
            neg_sample = unseen_rows.sample(1)
            neg_label = 0.0

        neg_idx = neg_sample.index.values[0]
        neg_item = neg_sample[self.item_col].values[0]

        label_value = 1.0 if pos_label > neg_label else -1.0

        pos_item_metadata = []
        neg_item_metadata = []
        if self.item_metadata is not None:
            pos_item_metadata = self.item_metadata[idx]
            neg_item_metadata = self.item_metadata[neg_idx]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item, dtype=torch.long),
            "pos_item_metadata": torch.as_tensor(pos_item_metadata),
            "neg_item_id": torch.tensor(neg_item, dtype=torch.long),
            "neg_item_metadata": torch.as_tensor(neg_item_metadata),
            "label": torch.as_tensor(
                label_value, dtype=torch.float32
            ),  # The MarginRankingLoss in PyTorch expects the target to be either 1 or -1, not arbitrary values.
        }

    @classmethod
    def get_default_loss_fn(cls):
        loss_fn = nn.MarginRankingLoss(margin=1.0)
        return loss_fn

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        pos_predictions, neg_predictions = model.predict_train_batch(batch_input)
        label = batch_input["label"].to(device)
        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()
        loss = loss_fn(pos_predictions, neg_predictions, label)
        return loss
