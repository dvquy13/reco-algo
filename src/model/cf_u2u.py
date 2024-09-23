from typing import Any, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from src.math_utils import sigmoid


class User2UserCollaborativeFiltering:
    """
    A class that implements a user-to-user collaborative filtering recommendation system.

    This system predicts the ratings or recommendations for a user by considering
    ratings from other similar users. The similarity between users is computed using
    cosine similarity on the user-item interaction matrix.

    Attributes:
        num_users (int): The number of users.
        num_items (int): The number of items.
        user_item_matrix (np.ndarray): A matrix that stores user-item interactions.
        user_similarity (np.ndarray): A matrix that stores user-user similarity scores.
    """

    def __init__(self, num_users: int, num_items: int):
        """
        Initializes the collaborative filtering model with placeholders for the
        user-item interaction matrix and the user-user similarity matrix.

        Args:
            num_users (int): The number of users in the system.
            num_items (int): The number of items in the system.
        """
        self.num_users = num_users
        self.num_items = num_items

        # Placeholder for user-item matrix
        self.user_item_matrix = np.zeros((num_users, num_items))

        # Placeholder for similarity matrix
        self.user_similarity = np.zeros((num_users, num_users))

    def forward(
        self, user: int, item: int, top_n: int = 10, debug: bool = False
    ) -> float:
        """
        Predict the rating for a given user-item pair by considering the top_n most similar users.

        Args:
            user (int): The user ID for whom the prediction is to be made.
            item (int): The item ID for which the rating is to be predicted.
            top_n (int): The number of most similar users to consider for prediction. Default is 10.
            debug (bool): If True, a debugger will be invoked during execution for debugging purposes.

        Returns:
            float: The predicted rating for the user-item pair.
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

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        """
        Fit the collaborative filtering model by constructing the user-item matrix
        and computing the user-user similarity matrix using cosine similarity.

        Args:
            user_ids (np.ndarray): A list or array of user IDs.
            item_ids (np.ndarray): A list or array of item IDs corresponding to the user IDs.
            ratings (np.ndarray): A list or array of ratings corresponding to the user-item pairs.
        """
        # Step 1: Create user-item matrix
        for user, item, rating in zip(user_ids, item_ids, ratings):
            self.user_item_matrix[user, item] = rating

        # Step 2: Compute user-user similarity using cosine similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)

        # Zero out self-similarity (diagonal of the matrix)
        np.fill_diagonal(self.user_similarity, 0)

    def predict(
        self, users: np.ndarray, items: np.ndarray, top_n: int = 10
    ) -> np.ndarray:
        """
        Predict the interaction scores (ratings) for a list of user-item pairs, considering top_n neighbors.

        Args:
            users (np.ndarray): A list or array of user IDs.
            items (np.ndarray): A list or array of item IDs corresponding to the users.
            top_n (int): The number of most similar users to consider for each prediction.

        Returns:
            np.ndarray: An array of predicted ratings for each user-item pair.
        """
        users = np.asarray(users)
        items = np.asarray(items)

        predictions = np.array(
            [self.forward(user, item, top_n=top_n) for user, item in zip(users, items)]
        )

        return sigmoid(predictions)

    def recommend(self, users: np.ndarray, k: int, top_n: int = 10) -> Dict[str, Any]:
        """
        Generate top-k item recommendations for each user in the provided list.

        Args:
            users (np.ndarray): A list or array of user IDs.
            k (int): The number of items to recommend for each user.
            top_n (int): The number of most similar users to consider for each prediction.

        Returns:
            Dict[str, Any]: A dictionary with user indices, recommended items, and their predicted scores.
        """
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
