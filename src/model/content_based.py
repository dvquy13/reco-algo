from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class ContentBased:
    def __init__(
        self,
        item_features: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the ContentBased.

        Args:
            item_features (torch.Tensor): Tensor of shape (num_items, feature_dim) representing item features.
            device (torch.device, optional): Device to perform computations on. Defaults to CPU.
        """
        self.device = device if device is not None else torch.device("cpu")
        self.item_features = torch.as_tensor(item_features).to(self.device)

        # Normalize item features to unit vectors for cosine similarity
        self.normalized_item_features = F.normalize(self.item_features, p=2, dim=1)

    def forward(self, item1: int, item2: int) -> torch.Tensor:
        """
        Compute similarity score between two items.

        Args:
            item1 (int): First item ID.
            item2 (int): Second item ID.

        Returns:
            torch.Tensor: Similarity score.
        """
        # Retrieve normalized feature vectors
        feature1 = self.normalized_item_features[item1]  # Shape: (feature_dim)
        feature2 = self.normalized_item_features[item2]  # Shape: (feature_dim)

        # Compute cosine similarity
        similarity = torch.dot(feature1, feature2)

        return similarity

    def predict(self, item1: List[int], item2: List[int]) -> torch.Tensor:
        """
        Compute similarity scores between pairs of items.

        Args:
            item1 (List[int]): List of first item indices.
            item2 (List[int]): List of second item indices.

        Returns:
            torch.Tensor: Tensor of similarity scores.
        """
        item1_tensor = torch.tensor(item1, dtype=torch.long).to(self.device)
        item2_tensor = torch.tensor(item2, dtype=torch.long).to(self.device)

        # Retrieve normalized feature vectors
        features1 = self.normalized_item_features[
            item1_tensor
        ]  # Shape: (batch_size, feature_dim)
        features2 = self.normalized_item_features[
            item2_tensor
        ]  # Shape: (batch_size, feature_dim)

        # Compute cosine similarity
        similarity = torch.sum(features1 * features2, dim=1)  # Shape: (batch_size)

        return similarity

    def recommend(
        self, users: List[int], items: List[int], k: int
    ) -> Dict[str, List[Any]]:
        """
        Generate top k item recommendations with scores for each input item in the provided list.

        Parameters:
        - items: List or array of item IDs for which to generate recommendations.
        - k: Number of top recommendations to return for each item.

        Returns:
        - recommendations_dict: A flattened dictionary containing:
            {
                "user_indice": [user1, user1, user2, user2, ...],
                "recommendation": [similar_item1, similar_item2, similar_item3, similar_item4 ...],
                "score": [score1, score2, score3, score4 ...]
            }
        """

        assert len(users) == len(items)

        user_indices = []
        all_recommendations = []
        all_scores = []

        iterable = tqdm(
            enumerate(items), desc="Generating Recommendations", total=len(users)
        )
        for i, item in iterable:
            # Check if the item exists
            if item < 0 or item >= self.normalized_item_features.size(0):
                print(f"Item ID {item} is out of bounds. Skipping.")
                continue

            # Compute similarities between the current item and all other items
            current_item_feature = self.normalized_item_features[item].unsqueeze(
                0
            )  # Shape: (1, feature_dim)
            similarities = torch.mm(
                current_item_feature, self.normalized_item_features.t()
            ).squeeze(
                0
            )  # Shape: (num_items)

            # Exclude the current item from recommendations by setting its similarity to -inf
            similarities[item] = -float("inf")

            # Get top k similar items
            top_k_scores, top_k_indices = torch.topk(
                similarities, min(k, similarities.size(0) - 1)
            )

            # Convert to lists
            top_k_scores = top_k_scores.cpu().tolist()
            top_k_indices = top_k_indices.cpu().tolist()

            # Append to the flattened lists
            user = users[i]
            user_indices.extend([user] * len(top_k_indices))
            all_recommendations.extend(top_k_indices)
            all_scores.extend(top_k_scores)

        # Assemble the final dictionary
        recommendations_dict = {
            "user_indice": user_indices,
            "recommendation": all_recommendations,
            "score": all_scores,
        }

        return recommendations_dict

    @classmethod
    def get_expected_dataset_type(cls):
        return None  # Updated to None since no dataset is used
