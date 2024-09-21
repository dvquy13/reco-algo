from typing import Any, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.math_utils import sigmoid


class User2UserCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

        # Placeholder for user-item matrix
        self.user_item_matrix = np.zeros((num_users, num_items))

        # Placeholder for similarity matrix
        self.user_similarity = np.zeros((num_users, num_users))

    def forward(self, user, item, top_n=10, debug=False):
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

        if debug:
            import pdb

            pdb.set_trace()

        if len(sim_scores) == 0 or sim_scores.sum() == 0:
            return 0

        # Step 1: Find the top N most similar users
        if len(sim_scores) > top_n:
            # Get indices of the top_n most similar users
            top_n_indices = np.argsort(sim_scores)[-top_n:]
            sim_scores = sim_scores[top_n_indices]
            user_ratings = user_ratings[top_n_indices]

        # Step 2: Compute the weighted average of ratings from top N similar users
        output = np.dot(sim_scores, user_ratings) / np.sum(sim_scores)
        return output

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

        return sigmoid(predictions)

    def recommend(
        self,
        users: np.ndarray,
        k: int,
        top_n: int = 10,
        progress_bar_type: str = "tqdm",
    ) -> Dict[str, Any]:
        """
        Generate top k item recommendations with scores for each user in the provided list.

        Parameters:
        - users: List or array of user IDs for whom to generate recommendations.
        - k: Number of top recommendations to return for each user.
        - top_n: Number of top similar users to consider when predicting scores.
        - progress_bar_type: Type of progress bar to use ('tqdm' or 'tqdm_notebook').

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indice": [user1, user1, user2, user2, ...],
                "recommendation": [item1, item2, item3, item4, ...],
                "score": [score1, score2, score3, score4, ...]
            }
        """
        if progress_bar_type == "tqdm_notebook":
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm

        if type(users) == list:
            users = np.array(users)

        all_items = np.arange(self.num_items)

        user_indices = []
        recommendations = []
        scores = []

        total_users = len(users)
        for i in tqdm(range(0, total_users), desc="Generating recommendations"):
            user = users[i]

            # Predict scores for all items for the current user
            predicted_scores = self.predict(
                [user] * len(all_items), all_items, top_n=top_n
            )

            # Get the top k items with the highest scores
            top_k_items = np.argsort(predicted_scores)[-k:][::-1]
            top_k_scores = predicted_scores[top_k_items]

            # Store results
            user_indices.extend([user] * k)
            recommendations.extend(top_k_items)
            scores.extend(top_k_scores)

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }
