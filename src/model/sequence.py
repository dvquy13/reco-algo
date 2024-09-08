import torch
import torch.nn as nn

from src.dataset_loader import ItemSequenceDataset


class SequenceRec(nn.Module):
    def __init__(self, num_items, embedding_dim, device="cpu"):
        super(SequenceRec, self).__init__()
        self.device = device

        # Add 1 to num_items for the unknown item (-1 padding)
        self.embedding = nn.Embedding(
            num_items + 1,  # One additional index for unknown/padding item
            embedding_dim,
            padding_idx=num_items,  # The additional index for the unknown item
        ).to(self.device)

        # Fully connected layer to map mean pooling output + target embedding to rating prediction
        self.fc_rating = nn.Linear(
            embedding_dim * 2, 1  # Concatenating sequence and target embeddings
        ).to(self.device)

    def forward(self, input_seq, target_item):
        # Ensure input tensors are on the correct device
        input_seq = torch.as_tensor(input_seq).to(self.device)
        target_item = torch.as_tensor(target_item).to(self.device)

        # Replace -1 in input_seq and target_item with num_items (padding_idx)
        padding_idx_tensor = torch.tensor(self.embedding.padding_idx).to(self.device)
        input_seq = torch.where(input_seq == -1, padding_idx_tensor, input_seq)
        target_item = torch.where(target_item == -1, padding_idx_tensor, target_item)

        # Embed input sequence
        embedded_seq = self.embedding(input_seq)

        # Mean pooling: take the mean over the sequence dimension (dim=1)
        pooled_output = embedded_seq.mean(dim=1)

        # Embed the target item
        embedded_target = self.embedding(target_item)

        # Concatenate the pooled sequence output with the target item embedding
        combined_embedding = torch.cat((pooled_output, embedded_target), dim=1)

        # Project combined embedding to rating prediction
        output_ratings = self.fc_rating(combined_embedding)

        return output_ratings

    def predict(self, input_seq, target_item):
        # Forward pass to get rating prediction
        output_ratings = self.forward(input_seq, target_item)

        # Return the predicted rating
        return output_ratings

    def predict_train_batch(self, batch_input: dict):
        item_sequence = batch_input["item_sequence"].to(self.device)
        target = batch_input["target"].to(self.device)

        predictions = self.predict(item_sequence, target)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return ItemSequenceDataset
