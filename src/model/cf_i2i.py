import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from src.math_utils import sigmoid


class Item2ItemCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

        # Placeholder for user-item matrix
        self.user_item_matrix = np.zeros((num_users, num_items))

        # Placeholder for item similarity matrix
        self.item_similarity = np.zeros((num_items, num_items))

    def forward(self, user, item, top_n=10, debug=False):
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

        if debug:
            import pdb

            pdb.set_trace()

        if len(sim_scores) == 0 or sim_scores.sum() == 0:
            return 3  # If no similar items are rated by the user, return neutral rating

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

        return sigmoid(predictions)

    def recommend(self, users, k, top_n=10, progress_bar_type="tqdm"):
        """
        Generate top k item recommendations with scores for each user in the provided list.

        Parameters:
        - users: List or array of user IDs for whom to generate recommendations.
        - k: Number of top recommendations to return for each user.
        - top_n: Number of top similar items to consider when predicting scores.
        - progress_bar_type: Type of progress bar to use ('tqdm' or 'tqdm_notebook').

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indice": [user1, user1, user2, user2, ...],
                "recommendation": [item1, item2, item3, item4, ...],
                "score": [score1, score2, score3, score4, ...]
            }
        """
        # Select the appropriate tqdm function based on progress_bar_type
        if progress_bar_type == "tqdm":
            progress_bar = tqdm
        elif progress_bar_type == "tqdm_notebook":
            progress_bar = tqdm_notebook
        else:
            raise ValueError(
                "progress_bar_type must be either 'tqdm' or 'tqdm_notebook'"
            )

        user_indices = []
        all_recommendations = []
        all_scores = []

        for user in progress_bar(users, desc="Generating Recommendations"):
            # User's ratings vector
            user_ratings = self.user_item_matrix[user, :]  # shape (num_items,)
            rated_items_indices = np.where(user_ratings != 0)[0]
            unrated_items_indices = np.where(user_ratings == 0)[0]

            if len(unrated_items_indices) == 0:
                continue  # User has rated all items

            if len(rated_items_indices) == 0:
                continue  # User has not rated any items, cannot make predictions

            # Similarities between unrated items and rated items
            similarities = self.item_similarity[
                unrated_items_indices[:, np.newaxis], rated_items_indices
            ]  # shape (num_unrated_items, num_rated_items)

            # For each unrated item, get the top_n similar rated items
            if similarities.shape[1] > top_n:
                top_n_sim_indices = np.argpartition(-similarities, top_n - 1, axis=1)[
                    :, :top_n
                ]
            else:
                top_n_sim_indices = np.argsort(-similarities, axis=1)

            # Gather the top_n similarities and ratings
            row_indices = np.arange(similarities.shape[0])[:, np.newaxis]
            top_n_similarities = similarities[row_indices, top_n_sim_indices]
            top_n_rated_item_indices = rated_items_indices[top_n_sim_indices]
            top_n_ratings = user_ratings[top_n_rated_item_indices]

            # Compute weighted sum for each unrated item
            numerator = np.sum(top_n_similarities * top_n_ratings, axis=1)
            denominator = np.sum(top_n_similarities, axis=1)

            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                predicted_ratings = numerator / denominator
                predicted_ratings[np.isnan(predicted_ratings)] = 0

            # Get the top k items
            if len(predicted_ratings) == 0:
                continue

            if len(predicted_ratings) > k:
                top_k_indices = np.argpartition(-predicted_ratings, k - 1)[:k]
            else:
                top_k_indices = np.argsort(-predicted_ratings)
            top_k_items = unrated_items_indices[top_k_indices]
            top_k_scores = predicted_ratings[top_k_indices]

            # Append to the flattened lists
            user_indices.extend([user] * len(top_k_items))
            all_recommendations.extend(top_k_items.tolist())
            all_scores.extend(top_k_scores.tolist())

        # Assemble the final dictionary
        recommendations_dict = {
            "user_indice": user_indices,
            "recommendation": all_recommendations,
            "score": all_scores,
        }

        return recommendations_dict
