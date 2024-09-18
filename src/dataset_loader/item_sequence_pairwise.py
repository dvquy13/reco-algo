import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ItemSequencePairwiseDataset(Dataset):
    PADDING_VALUE = -1

    def __init__(
        self,
        interaction_df,
        user_col,
        item_col,
        rating_col,
        timestamp_col,
        val_timestamp: int,
        is_train=True,
        max_input_sequence_length: int = 5,
        num_negative_samples: int = 5,
        item_metadata=None,
    ):
        # self.df is the full dataframe containing both train and val
        self.df = interaction_df.sort_values(timestamp_col, ascending=True).reset_index(
            drop=True
        )
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.max_input_sequence_length = max_input_sequence_length
        self.num_negative_samples = num_negative_samples
        self.item_metadata = item_metadata

        # Split the data into training or validation set based on timestamp
        if is_train:
            self.iteration_df = self.df.loc[
                lambda df: df[timestamp_col].lt(val_timestamp)
            ]
        else:
            self.iteration_df = self.df.loc[
                lambda df: df[timestamp_col].ge(val_timestamp)
            ]

        # Group data by user_id for easier sampling
        self.full_user_groups = self.df.groupby(user_col)
        if is_train:
            self.iteration_user_groups = self.iteration_df.groupby(user_col)
        else:
            self.iteration_user_groups = self.df.groupby(
                user_col
            )  # same as full_user_groups

        self.all_items = self.df[item_col].unique()

    def get_item_sequence(self, user, timestamp):
        # Get the item sequence for the user before the current timestamp
        item_sequence = (
            self.df.loc[
                lambda df: df[self.user_col].eq(user)
                & df[self.timestamp_col].lt(timestamp)
            ][self.item_col]
            .tail(self.max_input_sequence_length)  # Get the most recent items
            .values
        )

        # Pad the sequence if it's shorter than max_input_sequence_length
        sequence_length = len(item_sequence)
        if sequence_length < self.max_input_sequence_length:
            padding_needed = self.max_input_sequence_length - sequence_length
            item_sequence = np.pad(
                item_sequence,
                (padding_needed, 0),  # Add padding at the beginning
                "constant",
                constant_values=self.PADDING_VALUE,
            )

        return item_sequence

    def __len__(self):
        return len(self.iteration_df)

    def __getitem__(self, idx):
        row = self.iteration_df.iloc[idx]
        user = row[self.user_col]
        pos_item = row[self.item_col]
        pos_label = row[self.rating_col]
        timestamp = row[self.timestamp_col]
        item_sequence = self.get_item_sequence(user, timestamp)

        iteration_user_group = self.iteration_user_groups.get_group(user)
        full_user_group = self.full_user_groups.get_group(user)

        # Get all items the user has interacted with (in full dataset)
        user_items = full_user_group[self.item_col].unique()

        # Get negative samples from user interactions where rating < pos_label
        user_neg_samples = iteration_user_group.loc[
            lambda df: df[self.timestamp_col].lt(timestamp)
            & df[self.rating_col].lt(pos_label)
        ]

        neg_items_from_user = user_neg_samples[self.item_col].unique()
        neg_items = []
        neg_labels = []
        num_neg_samples_needed = self.num_negative_samples

        if len(neg_items_from_user) >= num_neg_samples_needed:
            sampled_neg_items = np.random.choice(
                neg_items_from_user, num_neg_samples_needed, replace=False
            ).tolist()
            neg_items.extend(sampled_neg_items)
            neg_labels.extend(
                user_neg_samples.loc[
                    user_neg_samples[self.item_col].isin(sampled_neg_items),
                    self.rating_col,
                ].tolist()
            )
        else:
            # Use all available negative samples from user interactions
            neg_items.extend(neg_items_from_user.tolist())
            neg_labels.extend(user_neg_samples[self.rating_col].tolist())
            num_neg_samples_needed -= len(neg_items_from_user)

            # Sample additional negative items from unseen items
            unseen_items = np.setdiff1d(self.all_items, user_items)
            additional_neg_items = np.random.choice(
                unseen_items, num_neg_samples_needed, replace=False
            ).tolist()
            neg_items.extend(additional_neg_items)
            neg_labels.extend([0.0] * len(additional_neg_items))

        # Compute labels for MarginRankingLoss
        labels = np.array([pos_label - nl for nl in neg_labels])

        # Retrieve item metadata if available
        pos_item_metadata = []
        neg_item_metadata = []
        if self.item_metadata is not None:
            pos_item_metadata = self.item_metadata.get(pos_item, [])
            neg_item_metadata = [self.item_metadata.get(item, []) for item in neg_items]

        return {
            self.user_col: torch.tensor(user, dtype=torch.long),
            "item_sequence": torch.tensor(item_sequence, dtype=torch.long),
            "target": torch.tensor(pos_item, dtype=torch.long),
            "rating": torch.tensor(pos_label, dtype=torch.float32),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "pos_item_metadata": torch.tensor(pos_item_metadata, dtype=torch.float32),
            "neg_item_metadata": torch.tensor(neg_item_metadata, dtype=torch.float32),
        }

    @classmethod
    def get_default_loss_fn(cls):
        return nn.MarginRankingLoss(margin=1.0)

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        item_sequences = batch_input["item_sequence"].to(device)
        target_items = batch_input["target"].to(device)
        neg_items = batch_input["neg_items"].to(device)
        labels = batch_input["labels"].to(device)

        # Ensure that neg_items and labels are 2D (batch_size x num_negative_samples)
        if neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # Model predicts scores for the target items based on item sequences
        pos_predictions = model.predict(
            item_sequences, target_items
        )  # shape [batch_size, 1]

        # Expand pos_predictions to match the shape of neg_predictions
        num_neg_samples = neg_items.shape[1]
        pos_predictions_expanded = pos_predictions.expand(-1, num_neg_samples).reshape(
            -1
        )

        # Flatten neg_items for batch prediction
        item_sequences_expanded = item_sequences.unsqueeze(1).expand(
            -1, num_neg_samples, -1
        )
        item_sequences_flat = item_sequences_expanded.reshape(
            -1, item_sequences.shape[-1]
        )
        neg_items_flat = neg_items.reshape(-1)

        neg_predictions_flat = model.predict(item_sequences_flat, neg_items_flat)

        neg_predictions_flat = neg_predictions_flat.reshape(-1)
        labels_flat = labels.reshape(-1)

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()

        loss = loss_fn(pos_predictions_expanded, neg_predictions_flat, labels_flat)

        return loss
