import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDataset


class WideAndDeep(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_units, device):
        super(WideAndDeep, self).__init__()

        self.device = device

        # Wide part (memorization) - simple linear interaction between user and item indices
        self.user_bias = nn.Embedding(num_users, 1).to(self.device)
        self.item_bias = nn.Embedding(num_items, 1).to(self.device)

        # Deep part (generalization) - embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)

        # Fully connected layers for the deep part
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_units).to(self.device)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2).to(self.device)
        self.fc3 = nn.Linear(hidden_units // 2, 1).to(self.device)

        # Activation and dropout
        self.relu = nn.ReLU().to(self.device)
        self.dropout = nn.Dropout(0.2).to(self.device)

    def forward(self, user, item):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)

        # Wide part: user and item biases (for memorization)
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()
        wide_output = user_bias + item_bias

        # Deep part: user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through the deep layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        deep_output = self.fc3(x).squeeze()

        # Combine wide and deep parts
        output = wide_output + deep_output
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
