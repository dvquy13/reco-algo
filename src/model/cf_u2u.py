import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class User2UserCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

        # Placeholder for user-item matrix
        self.user_item_matrix = np.zeros((num_users, num_items))

        # Placeholder for similarity matrix
        self.user_similarity = np.zeros((num_users, num_users))

    def forward(self, user, item, top_n=10):
        """
        Predict rating for a given user-item pair considering only top_n most similar neighbors.
        """
        # Compute prediction using weighted average of ratings from similar users
        sim_scores = self.user_similarity[user]
        user_ratings = self.user_item_matrix[:, item]

        # Only consider users who have rated the item
        rated_mask = user_ratings != 0
        sim_scores = sim_scores[rated_mask]
        user_ratings = user_ratings[rated_mask]

        if len(sim_scores) == 0 or sim_scores.sum() == 0:
            return 0  # If no users have rated the item, return 0

        # Step 1: Find the top N most similar users
        if len(sim_scores) > top_n:
            # Get indices of the top_n most similar users
            top_n_indices = np.argsort(sim_scores)[-top_n:]
            sim_scores = sim_scores[top_n_indices]
            user_ratings = user_ratings[top_n_indices]

        # Step 2: Compute the weighted average of ratings from top N similar users
        return np.dot(sim_scores, user_ratings) / np.sum(sim_scores)

    def fit(self, user_ids, item_ids, ratings):
        """
        Fit the collaborative filtering model by constructing the user-item matrix
        and computing the user-user similarity matrix.
        """
        # Step 1: Create user-item matrix
        for user, item, rating in zip(user_ids, item_ids, ratings):
            self.user_item_matrix[user, item] = rating

        # Step 2: Compute user-user similarity using cosine similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)

        # Zero out self-similarity (diagonal of the matrix)
        np.fill_diagonal(self.user_similarity, 0)

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
