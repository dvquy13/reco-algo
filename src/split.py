from datetime import timedelta

from loguru import logger


def train_test_split_timebased(
    rating_score_df,
    val_num_days=7,
    user_col="user_id",
    item_col="item_id",
    timestamp_col="timestamp",
    remove_unseen_users_in_test=True,
    remove_unseen_items_in_test=True,
):
    max_date = rating_score_df[timestamp_col].max().date()
    val_date = max_date - timedelta(days=val_num_days)

    val_date_str = val_date.strftime("%Y-%m-%d")

    val_df = rating_score_df.loc[lambda df: df[timestamp_col].ge(val_date_str)]
    train_df = rating_score_df.loc[lambda df: df[timestamp_col].lt(val_date_str)]

    if remove_unseen_users_in_test:
        logger.info(f"Removing the new users in val set...")
        train_users = train_df[user_col].unique()
        val_users_original = val_df[user_col].nunique()
        val_df = val_df.loc[lambda df: df[user_col].isin(train_users)]
        logger.info(
            f"Removed {val_users_original - val_df[user_col].nunique()} users from val set"
        )

    if remove_unseen_items_in_test:
        logger.info(f"Removing the new items in val set...")
        train_items = train_df[item_col].unique()
        val_items_original = val_df[item_col].nunique()
        val_df = val_df.loc[lambda df: df[item_col].isin(train_items)]
        logger.info(
            f"Removed {val_items_original - val_df[item_col].nunique()} items from val set"
        )

    return train_df, val_df
