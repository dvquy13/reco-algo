from torch.utils.data import Dataset


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
        return dict(user=user, item=item, rating=rating, item_metadata=item_metadata)
