import torch
import torch.nn as nn

from src.dataset_loader import UserItemRatingDataset


class LinearRegressionRatingPredictor(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        metadata_feature_size,
        device,
        dropout_prob=0.3,
    ):
        super(LinearRegressionRatingPredictor, self).__init__()

        self.device = device

        # Embeddings for users and items with specified embedding dimension
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_prob)

        # Global bias term (mean rating)
        self.global_bias = nn.Parameter(torch.tensor([0.0])).to(self.device)

        # Linear layer to learn weights for metadata features
        self.metadata_layer = nn.Linear(metadata_feature_size, 1).to(self.device)

        # Linear layer for combining user and item embeddings into a single score
        self.embedding_combiner = nn.Linear(embedding_dim, 1).to(self.device)

    def forward(self, user, item, item_metadata):
        # Move input tensors to the correct device
        user = user.to(self.device)
        item = item.to(self.device)
        item_metadata = item_metadata.to(self.device)

        # Get user and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # Apply dropout to the embeddings to regularize
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        # Combine user and item embeddings using a linear layer
        combined_embedding = self.embedding_combiner(user_emb * item_emb).squeeze()

        # Pass item metadata through a linear layer
        metadata_contribution = self.metadata_layer(item_metadata).squeeze()
        metadata_contribution = self.dropout(metadata_contribution)

        # Sum of combined user-item embedding, metadata contribution, and global bias
        output = combined_embedding + metadata_contribution + self.global_bias

        return output

    def predict(self, users, items, item_metadata):
        """
        Predict interaction score (rating) for given users and items.
        """
        users = torch.as_tensor(users)
        items = torch.as_tensor(items)
        item_metadata = torch.as_tensor(item_metadata)
        return self.forward(users, items, item_metadata)

    def predict_train_batch(self, batch_input: dict, device: str = "cpu"):
        users = batch_input["user"]
        items = batch_input["item"]
        item_metadata = batch_input["item_metadata"]

        users = users.to(device)
        items = items.to(device)
        item_metadata = item_metadata.to(device)

        predictions = self.predict(users, items, item_metadata)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return UserItemRatingDataset
