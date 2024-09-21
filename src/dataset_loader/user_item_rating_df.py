import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class UserItemRatingDFDataset(Dataset):
    def __init__(
        self,
        df,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        item_feature=None,
    ):
        self.df = df.assign(**{rating_col: df[rating_col].astype(np.float32)})
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.item_feature = item_feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df[self.user_col].iloc[idx]
        item = self.df[self.item_col].iloc[idx]
        rating = self.df[self.rating_col].iloc[idx]
        item_sequence = []
        if "item_sequence" in self.df:
            item_sequence = self.df["item_sequence"].iloc[idx]
        item_feature = []
        if self.item_feature is not None:
            item_feature = self.item_feature[idx]
        return dict(
            user=torch.as_tensor(user),
            item=torch.as_tensor(item),
            rating=torch.as_tensor(rating),
            item_sequence=torch.tensor(item_sequence, dtype=torch.long),
            item_feature=(
                torch.as_tensor(item_feature) if item_feature is not None else []
            ),
        )

    @classmethod
    def get_default_loss_fn(cls):
        loss_fn = nn.MSELoss()
        return loss_fn

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        predictions = model.predict_train_batch(batch_input, device=device).squeeze()
        ratings = batch_input["rating"].to(device).squeeze()

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()
        loss = loss_fn(predictions, ratings)
        return loss
