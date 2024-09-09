import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDataset


class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_units, device):
        super(TwoTower, self).__init__()

        self.device = device

        # User Tower: embedding and MLP for user features
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.user_fc1 = nn.Linear(embedding_dim, hidden_units).to(self.device)
        self.user_fc2 = nn.Linear(hidden_units, hidden_units // 2).to(self.device)

        # Item Tower: embedding and MLP for item features
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)
        self.item_fc1 = nn.Linear(embedding_dim, hidden_units).to(self.device)
        self.item_fc2 = nn.Linear(hidden_units, hidden_units // 2).to(self.device)

        # Activation and dropout
        self.relu = nn.ReLU().to(self.device)
        self.dropout = nn.Dropout(0.2).to(self.device)

    def forward(self, user, item):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)

        # User Tower
        user_emb = self.user_embedding(user)
        user_x = self.user_fc1(user_emb)
        user_x = self.relu(user_x)
        user_x = self.dropout(user_x)
        user_x = self.user_fc2(user_x)
        user_x = self.relu(user_x)

        # Item Tower
        item_emb = self.item_embedding(item)
        item_x = self.item_fc1(item_emb)
        item_x = self.relu(item_x)
        item_x = self.dropout(item_x)
        item_x = self.item_fc2(item_x)
        item_x = self.relu(item_x)

        # Final output is the dot product of the two towers' outputs
        output = torch.sum(user_x * item_x, dim=-1)

        return output

    def predict(self, users, items):
        """
        Predict interaction score for given users and items.
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
