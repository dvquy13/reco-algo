import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from src.math_utils import sigmoid


class Item2ItemCollaborativeFiltering:
    """
    Item-to-Item Collaborative Filtering model that recommends items to users
    based on item similarity and past user interactions (ratings). It computes
    item similarity using cosine similarity and predicts ratings for unrated items.
    """

    def __init__(self, num_users, num_items):
        """
        Initialize the Item2ItemCollaborativeFiltering model.

        Parameters:
        - num_users (int): The number of users in the dataset.
        - num_items (int): The number of items in the dataset.
        """
        self.num_users = num_users
        self.num_items = num_items

        # Placeholder for user-item matrix
        self.user_item_matrix = np.zeros((num_users, num_items))

        # Placeholder for item similarity matrix
        self.item_similarity = np.zeros((num_items, num_items))

    def forward(self, user, item, top_n=10, debug=False):
        """
        Predict the rating for a given user-item pair, considering the top_n most similar items.

        Parameters:
        - user (int): The index of the user.
        - item (int): The index of the item.
        - top_n (int): The number of top similar items to consider for prediction. Default is 10.
        - debug (bool): If True, triggers a debugger to inspect values.

        Returns:
        - float: The predicted rating for the user-item pair.
                 If no similar items are rated by the user, it returns 0.
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
            return 0  # No similar items rated by the user

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
        Fit the model by constructing the user-item matrix and computing the item-item similarity matrix.

        Parameters:
        - user_ids (array-like): List or array of user indices.
        - item_ids (array-like): List or array of item indices.
        - ratings (array-like): List or array of corresponding ratings.
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
        Predict interaction scores (ratings) for given users and items.

        Parameters:
        - users (array-like): List or array of user indices.
        - items (array-like): List or array of item indices.
        - top_n (int): Number of top similar items to consider when predicting scores. Default is 10.

        Returns:
        - np.ndarray: Array of predicted ratings.
        """
        users = np.asarray(users)
        items = np.asarray(items)

        predictions = np.array(
            [self.forward(user, item, top_n=top_n) for user, item in zip(users, items)]
        )

        return sigmoid(predictions)

    def recommend(self, users, k, top_n=10):
        """
        Generate top-k item recommendations with predicted scores for each user.

        Parameters:
        - users (array-like): List or array of user indices for whom to generate recommendations.
        - k (int): Number of top recommendations to return for each user.
        - top_n (int): Number of top similar items to consider when predicting scores. Default is 10.

        Returns:
        - dict: Dictionary with the following keys:
            - 'user_indice' (list): List of user indices for whom recommendations are generated.
            - 'recommendation' (list): List of recommended item indices.
            - 'score' (list): List of corresponding predicted scores for the recommended items.
        """
        user_indices = []
        all_recommendations = []
        all_scores = []

        for user in tqdm(users, desc="Generating Recommendations"):
            # User's ratings vector
            user_ratings = self.user_item_matrix[user, :]  # shape (num_items,)
            rated_items_indices = np.where(user_ratings != 0)[0]
            unrated_items_indices = np.where(user_ratings == 0)[0]

            if len(unrated_items_indices) == 0:
                continue  # User has rated all items

            if len(rated_items_indices) == 0:
                continue  # User has not rated any items, cannot make predictions

            # Similarities between unrated items and rated items
            # Find the subset of the item_similarity matrix where the rows are unrated items and
            # the columns are rated items. Values are how similar they are.
            similarities = self.item_similarity[
                unrated_items_indices[:, np.newaxis], rated_items_indices
            ]  # shape (num_unrated_items, num_rated_items)

            # For each unrated item, get the top_n similar rated items
            if similarities.shape[1] > top_n:
                # First use argpartition to find the top k similar (unsorted) between the
                # unrated items and rated items
                # After the argpartition step, the first k columns will contain the indices
                # of the rated items that are most similar to the unrated items in the corresponding rows
                # Then we can just take the first k columns.
                top_n_sim_indices = np.argpartition(-similarities, top_n - 1, axis=1)[
                    :, :top_n
                ]
            else:
                top_n_sim_indices = np.argsort(-similarities, axis=1)

            # Gather the top_n similarities and ratings
            # Build the empty rows corresponding to the number of unrated items
            row_indices = np.arange(similarities.shape[0])[:, np.newaxis]
            # Fill in the similarities with other items for those rows, but keep only the top similar (unsorted)
            top_n_similarities = similarities[row_indices, top_n_sim_indices]
            # Since rated_items_indices denote the indices of the rated items in [n,] shape
            # If we get the index from that vector using a matrix then it would return
            # a new matrix of the same shape but with the values of the original matrix
            # by the corresponding indices
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
            top_k_scores = sigmoid(predicted_ratings[top_k_indices])

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
