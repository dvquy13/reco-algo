import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDataset


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, device):
        super(MatrixFactorization, self).__init__()

        self.device = device

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)

    def forward(self, user, item):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)

        # Get user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # Dot product of user and item embeddings to get the predicted rating
        output = torch.sum(user_emb * item_emb, dim=-1)

        return output

    def predict(self, users, items):
        """
        Predict interaction score (rating) for given users and items.
        """
        users = torch.as_tensor(users)
        items = torch.as_tensor(items)
        return self.forward(users, items)

    def predict_train_batch(self, batch_input: dict, device: str = "cpu"):
        users = batch_input["user"]
        items = batch_input["item"]

        users = users.to(device)
        items = items.to(device)

        predictions = self.predict(users, items)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingDataset
