import torch
import torch.nn as nn

from src.dataset_loader import ItemSequenceDataset, ItemSequencePairwiseDataset


class SequencePairwiseRanking(nn.Module):
    def __init__(
        self, num_items, embedding_dim, device="cpu", max_input_sequence_length=5
    ):
        super(SequencePairwiseRanking, self).__init__()
        self.device = device
        self.max_input_sequence_length = max_input_sequence_length
        self.num_items = num_items

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
        embedded_seq = self.embedding(
            input_seq
        )  # Shape: [batch_size, seq_len, embedding_dim]

        # Mean pooling: take the mean over the sequence dimension (dim=1)
        pooled_output = embedded_seq.mean(dim=1)  # Shape: [batch_size, embedding_dim]

        # Embed the target item
        embedded_target = self.embedding(
            target_item
        )  # Shape: [batch_size, embedding_dim]

        # Concatenate the pooled sequence output with the target item embedding
        combined_embedding = torch.cat(
            (pooled_output, embedded_target), dim=1
        )  # Shape: [batch_size, embedding_dim*2]

        # Project combined embedding to rating prediction
        output_ratings = self.fc_rating(combined_embedding)  # Shape: [batch_size, 1]

        return output_ratings  # Shape: [batch_size]

    def predict(self, input_seq, target_item):
        # Forward pass to get rating prediction
        output_ratings = self.forward(input_seq, target_item)

        # Return the predicted rating
        return output_ratings

    def predict_train_batch(self, batch_input: dict):
        """
        Returns positive and negative predictions for a batch during training.

        Parameters:
        - batch_input: A dictionary containing batch data from ItemSequencePairwiseDataset.

        Returns:
        - pos_predictions: Predictions for positive items.
        - neg_predictions: Predictions for negative items.
        """
        item_sequences = batch_input["item_sequence"].to(self.device)
        target_items = batch_input["target"].to(self.device)
        neg_items = batch_input["neg_items"].to(self.device)

        # Ensure that neg_items is 2D (batch_size x num_negative_samples)
        if neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)

        batch_size = item_sequences.size(0)
        num_neg_samples = neg_items.size(1)

        # Positive predictions
        pos_predictions = self.predict(
            item_sequences, target_items
        )  # Shape: [batch_size]

        # Prepare item sequences and negative items for batch prediction
        # Expand item_sequences to match the number of negative samples
        item_sequences_expanded = item_sequences.unsqueeze(1).expand(
            -1, num_neg_samples, -1
        )
        item_sequences_flat = item_sequences_expanded.reshape(
            -1, item_sequences.size(-1)
        )  # Shape: [batch_size * num_neg_samples, seq_len]

        neg_items_flat = neg_items.reshape(-1)  # Shape: [batch_size * num_neg_samples]

        # Negative predictions
        neg_predictions_flat = self.predict(
            item_sequences_flat, neg_items_flat
        )  # Shape: [batch_size * num_neg_samples]

        # Reshape neg_predictions back to [batch_size, num_neg_samples]
        neg_predictions = neg_predictions_flat.view(
            batch_size, num_neg_samples
        )  # Shape: [batch_size, num_neg_samples]

        return pos_predictions, neg_predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return ItemSequencePairwiseDataset

    def recommend(
        self, users, k, user_item_sequences, batch_size=128, progress_bar_type="tqdm"
    ):
        """
        Generate top k item recommendations with scores for each user in the provided list.

        Parameters:
        - users: List or array of user IDs for whom to generate recommendations.
        - user_item_sequences: List or array of item sequences corresponding to each record in `users` input.
        - k: Number of top recommendations to return for each user.
        - batch_size: Batch size for processing users.
        - progress_bar_type: Type of progress bar to use ('tqdm' or 'tqdm_notebook').

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indice": [user1, user1, ..., user2, user2, ...],
                "recommendation": [item1, item2, ..., item1, item2, ...],
                "score": [score1, score2, ..., score1, score2, ...]
            }
        """
        if progress_bar_type == "tqdm_notebook":
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self.eval()  # Set model to evaluation mode
        all_items = torch.arange(self.num_items).to(self.device)  # Exclude padding_idx

        user_indices = []
        recommendations = []
        scores = []

        with torch.no_grad():
            total_users = len(users)
            # Split users and their sequences into batches
            for i in tqdm(
                range(0, total_users, batch_size), desc="Generating recommendations"
            ):
                user_batch = users[i : i + batch_size]
                sequence_batch = user_item_sequences[i : i + batch_size]

                # Prepare item sequences for the batch
                item_sequences_batch = []
                for item_sequence in sequence_batch:
                    # item_sequence is already an array or list
                    item_sequence = torch.tensor(item_sequence, dtype=torch.long)

                    # Pad or truncate the sequence
                    if len(item_sequence) < self.max_input_sequence_length:
                        padding_needed = self.max_input_sequence_length - len(
                            item_sequence
                        )
                        padding = torch.full(
                            (padding_needed,),
                            self.embedding.padding_idx,
                            dtype=torch.long,
                        )
                        item_sequence = torch.cat([padding, item_sequence], dim=0)
                    else:
                        item_sequence = item_sequence[-self.max_input_sequence_length :]

                    item_sequences_batch.append(item_sequence)

                # Stack item sequences into a tensor
                item_sequences_batch = torch.stack(item_sequences_batch).to(
                    self.device
                )  # Shape: [batch_size, seq_len]

                # Expand item sequences to match all items
                num_items = all_items.size(0)
                item_sequences_expanded = item_sequences_batch.unsqueeze(1).expand(
                    -1, num_items, -1
                )  # Shape: [batch_size, num_items, seq_len]
                item_sequences_flat = item_sequences_expanded.reshape(
                    -1, self.max_input_sequence_length
                )  # Shape: [batch_size * num_items, seq_len]

                # Expand all_items to match batch size
                target_items_expanded = (
                    all_items.unsqueeze(0)
                    .expand(len(user_batch), -1)
                    .reshape(-1)
                    .to(self.device)
                )  # Shape: [batch_size * num_items]

                # Predict scores for all items
                scores_flat = self.predict(
                    item_sequences_flat, target_items_expanded
                ).squeeze()  # Shape: [batch_size * num_items]

                # Reshape scores back to [batch_size, num_items]
                scores_batch = scores_flat.view(
                    len(user_batch), num_items
                )  # Shape: [batch_size, num_items]

                # Get top k items for each user in the batch
                topk_scores, topk_indices = torch.topk(scores_batch, k)
                topk_items = all_items[topk_indices]  # Shape: [batch_size, k]

                # Collect recommendations for the batch
                for idx, user in enumerate(user_batch):
                    user_indices.extend([user] * k)
                    recommendations.extend(topk_items[idx].cpu().numpy())
                    scores.extend(topk_scores[idx].cpu().numpy())

        recommendations_dict = {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }

        return recommendations_dict


class SequenceRatingPrediction(SequencePairwiseRanking):
    def predict_train_batch(self, batch_input: dict):
        item_sequence = batch_input["item_sequence"].to(self.device)
        target = batch_input["target"].to(self.device)

        predictions = self.predict(item_sequence, target)

        return predictions

    @classmethod
    def get_expected_dataset_type(cls):
        return ItemSequenceDataset
