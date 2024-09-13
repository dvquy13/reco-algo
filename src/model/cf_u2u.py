import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, tqdm_notebook


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
            return 3  # If no users have rated the item, return neutral rating

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

    def recommend(self, users, k, top_n=10, progress_bar_type="tqdm"):
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
                "user_indices": [user1, user1, user2, user2, ...],
                "recommendations": [item1, item2, item3, item4, ...],
                "scores": [score1, score2, score3, score4, ...]
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
            # Find items not yet rated by the user
            unrated_items = np.where(self.user_item_matrix[user, :] == 0)[0]

            if len(unrated_items) == 0:
                # If the user has rated all items, skip to next user
                continue

            # Predict scores for all unrated items
            predicted_scores = []
            for item in unrated_items:
                score = self.forward(user, item, top_n=top_n)
                score = float(score)
                predicted_scores.append((item, score))

            # Sort the predicted scores in descending order and select top k
            top_k = sorted(predicted_scores, key=lambda x: x[1], reverse=True)[:k]

            # Append to the flattened lists
            for item, score in top_k:
                user_indices.append(user)
                all_recommendations.append(item)
                all_scores.append(score)

        # Assemble the final dictionary
        recommendations_dict = {
            "user_indice": user_indices,
            "recommendation": all_recommendations,
            "score": all_scores,
        }

        return recommendations_dict
