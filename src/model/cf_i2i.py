import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Item2ItemCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

        # Placeholder for user-item matrix
        self.user_item_matrix = np.zeros((num_users, num_items))

        # Placeholder for item similarity matrix
        self.item_similarity = np.zeros((num_items, num_items))

    def forward(self, user, item, top_n=10):
        """
        Predict rating for a given user-item pair considering only top_n most similar items.
        """
        # Compute prediction using weighted average of ratings for similar items
        sim_scores = self.item_similarity[item]
        item_ratings = self.user_item_matrix[user, :]

        # Only consider items that have been rated by the user
        rated_mask = item_ratings != 0
        sim_scores = sim_scores[rated_mask]
        item_ratings = item_ratings[rated_mask]

        if len(sim_scores) == 0 or sim_scores.sum() == 0:
            return 0  # If no similar items are rated by the user, return 0

        # Step 1: Find the top N most similar items
        if len(sim_scores) > top_n:
            # Get indices of the top_n most similar items
            top_n_indices = np.argsort(sim_scores)[-top_n:]
            sim_scores = sim_scores[top_n_indices]
            item_ratings = item_ratings[top_n_indices]

        # Step 2: Compute the weighted average of ratings from top N similar items
        return np.dot(sim_scores, item_ratings) / np.sum(sim_scores)

    def fit(self, user_ids, item_ids, ratings):
        """
        Fit the collaborative filtering model by constructing the user-item matrix
        and computing the item-item similarity matrix.
        """
        # Step 1: Create user-item matrix
        for user, item, rating in zip(user_ids, item_ids, ratings):
            self.user_item_matrix[user, item] = rating

        # Step 2: Compute item-item similarity using cosine similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

        # Zero out self-similarity (diagonal of the matrix)
        np.fill_diagonal(self.item_similarity, 0)

    def predict(self, users, items, top_n=10):
        """
        Predict interaction score (rating) for given users and items considering top_n neighbors.
        """
        users = np.asarray(users)
        items = np.asarray(items)

        predictions = np.array(
            [self.forward(user, item, top_n=top_n) for user, item in zip(users, items)]
        )

        return predictions
