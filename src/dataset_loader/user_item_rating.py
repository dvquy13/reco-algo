import torch
from torch.utils.data import Dataset

from src.loss import mse_loss


class UserItemRatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings, item_metadata=None):
        """
        Args:
            user_ids (list or array): List of user indices.
            item_ids (list or array): List of item indices.
            ratings (list or array): List of corresponding ratings.
            item_metadata (2D array): Transformed item metadata matrix.
        """
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.item_metadata = item_metadata

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user = self.user_ids[idx]
        item = self.item_ids[idx]
        rating = self.ratings[idx]
        item_metadata = []
        if self.item_metadata is not None:
            item_metadata = self.item_metadata[idx]
        return dict(
            user=torch.as_tensor(user),
            item=torch.as_tensor(item),
            rating=torch.as_tensor(rating),
            item_metadata=torch.as_tensor(item_metadata) if item_metadata else [],
        )

    @classmethod
    def get_default_loss_fn(cls):
        loss_fn = mse_loss
        return loss_fn

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        predictions = model.predict_train_batch(batch_input, device=device)
        ratings = batch_input["rating"].to(device)

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()
        loss = loss_fn(predictions, ratings, l2_reg=None, model=model, device=device)
        return loss
