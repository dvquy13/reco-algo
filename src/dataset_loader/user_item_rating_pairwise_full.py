import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class UserItemRatingPairwiseFullDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        label_col: str,
        timestamp_col: str,
        val_timestamp: int,
        is_train=True,
        item_metadata=None,
        num_negative_samples=5,
    ):
        # df here refers to the full df containing both train and val
        # Reset index so that the index corresponds to the row number
        self.df = df.sort_values(timestamp_col, ascending=True).reset_index(drop=True)
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.item_metadata = item_metadata
        self.timestamp_col = timestamp_col
        self.num_negative_samples = num_negative_samples

        # iteration_df is specific to train or val set, but we want to keep track of the whole interaction_df to access the input sequence
        if is_train:
            self.iteration_df = self.df.loc[
                lambda df: df[timestamp_col].lt(val_timestamp)
            ]
        else:
            self.iteration_df = self.df.loc[
                lambda df: df[timestamp_col].gt(val_timestamp)
            ]

        # full_user_groups is to identify all the items user has interacted with in either train or val,
        # so that when negative sampling we don't get false negative
        self.full_user_groups = self.df.groupby(user_col)
        # iteration_user_groups is to identify the seen negative samples for a given user
        if is_train:
            self.iteration_user_groups = self.iteration_df.groupby(user_col)
        else:
            self.iteration_user_groups = self.df.groupby(user_col)

    def __len__(self):
        return len(self.iteration_df)

    def __getitem__(self, idx: int):
        # Get the data at the index
        row = self.iteration_df.iloc[idx]
        user_id = row[self.user_col]
        pos_item = row[self.item_col]
        pos_label = row[self.label_col]

        # Get a group of data for the same user
        iteration_user_group = self.iteration_user_groups.get_group(user_id)
        full_user_group = self.full_user_groups.get_group(user_id)

        # Find all items the user has interacted with
        user_items = set(full_user_group[self.item_col])

        # Get negative samples for the same user where label < pos_label
        user_neg_samples = iteration_user_group[
            iteration_user_group[self.label_col] < pos_label
        ]
        neg_indices = user_neg_samples.index.tolist()

        neg_sample_indices = []
        neg_labels = []
        num_neg_samples_needed = self.num_negative_samples

        if len(neg_indices) >= num_neg_samples_needed:
            sampled_neg_indices = np.random.choice(
                neg_indices, num_neg_samples_needed, replace=False
            ).tolist()
            neg_sample_indices.extend(sampled_neg_indices)
            neg_labels.extend(self.df.loc[sampled_neg_indices, self.label_col].tolist())
        else:
            # Use all available negative samples from user interactions
            neg_sample_indices.extend(neg_indices)
            neg_labels.extend(self.df.loc[neg_indices, self.label_col].tolist())
            num_neg_samples_needed -= len(neg_indices)

            # Sample additional negative items from unseen items
            unseen_indices = self.df[
                ~self.df[self.item_col].isin(user_items)
            ].index.tolist()
            additional_neg_indices = np.random.choice(
                unseen_indices, num_neg_samples_needed, replace=False
            ).tolist()
            neg_sample_indices.extend(additional_neg_indices)
            neg_labels.extend([0.0] * len(additional_neg_indices))

        # Retrieve negative items
        neg_items = self.df.loc[neg_sample_indices, self.item_col].values

        # Compute labels for MarginRankingLoss
        # labels = np.array([1.0 if pos_label > nl else -1.0 for nl in neg_labels])
        # Below is a version where we weight the label by the actual differences in ratings
        labels = np.array([pos_label - nl for nl in neg_labels])

        # Retrieve item metadata
        pos_item_metadata = []
        neg_item_metadata = []
        if self.item_metadata is not None:
            pos_item_metadata = self.item_metadata[idx]
            neg_item_metadata = [self.item_metadata[i] for i in neg_sample_indices]

        return {
            self.user_col: torch.tensor(user_id, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item, dtype=torch.long),
            "pos_item_metadata": torch.as_tensor(pos_item_metadata),
            "neg_item_ids": torch.tensor(neg_items, dtype=torch.long),
            "neg_item_metadata": torch.as_tensor(neg_item_metadata),
            "labels": torch.as_tensor(labels, dtype=torch.float32),
        }

    @classmethod
    def get_default_loss_fn(cls):
        return nn.MarginRankingLoss(margin=1.0)

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        # Use model's predict_train_batch to get positive and negative predictions
        pos_predictions, neg_predictions = model.predict_train_batch(batch_input)

        # Move labels to the device
        labels = batch_input["labels"].to(device)

        # Ensure labels and neg_predictions are flattened
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        labels_flat = labels.reshape(-1)

        neg_predictions_flat = neg_predictions.reshape(-1)

        # Expand pos_predictions to match the shape of neg_predictions_flat
        num_neg_samples = neg_predictions.shape[1]
        pos_predictions_expanded = pos_predictions.expand(-1, num_neg_samples).reshape(
            -1
        )

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()

        # Compute the loss
        loss = loss_fn(pos_predictions_expanded, neg_predictions_flat, labels_flat)

        return loss
