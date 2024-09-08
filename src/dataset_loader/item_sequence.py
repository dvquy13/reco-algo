import numpy as np
from torch.utils.data import Dataset


class ItemSequenceDataset(Dataset):
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
    ):
        self.interaction_df = interaction_df.sort_values(timestamp_col, ascending=True)
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.max_input_sequence_length = max_input_sequence_length
        # iteration_df is specific to train or val set, but we want to keep track of the whole interaction_df to access the input sequence
        if is_train:
            self.iteration_df = self.interaction_df.loc[
                lambda df: df[timestamp_col].lt(val_timestamp)
            ]
        else:
            self.iteration_df = self.interaction_df.loc[
                lambda df: df[timestamp_col].gt(val_timestamp)
            ]

    def get_item_sequence(self, user, timestamp):
        # Get the item sequence for the user before the current timestamp
        item_sequence = (
            self.interaction_df.loc[
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
        return self.iteration_df.shape[0]

    def __getitem__(self, idx):
        row = self.iteration_df.iloc[idx]
        user = row[self.user_col]
        item = row[self.item_col]
        rating = row[self.rating_col]
        timestamp = row[self.timestamp_col]
        item_sequence = self.get_item_sequence(user, timestamp)

        return dict(item_sequence=item_sequence, target=item, rating=rating)
