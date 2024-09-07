import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_layers,
        user_ids,
        item_ids,
        interaction_scores=None,
        dropout_rate=0.5,
        device="cpu",  # Added device argument with default to 'cpu'
    ):
        """
        LightGCN model
        Args:
            embedding_dim (int): Embedding dimension
            n_layers (int): Number of propagation layers
            user_ids (list or tensor): List of user indices
            item_ids (list or tensor): List of item indices
            interaction_scores (list or tensor, optional): Interaction scores (binary or weighted)
            device (str or torch.device, optional): The device to run computations on ('cpu' or 'cuda').
        """
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = torch.device(device)

        # Infer the number of users and items from the data
        self.n_users = max(user_ids) + 1  # Assuming user_ids are 0-indexed
        self.n_items = max(item_ids) + 1  # Assuming item_ids are 0-indexed

        # Initialize user and item embeddings using normal initialization
        self.user_embedding = nn.Embedding(self.n_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, embedding_dim).to(self.device)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # Build adjacency matrix from user-item interactions
        self.adj_matrix = self.build_adj_matrix(
            user_ids, item_ids, interaction_scores
        ).to(
            "cpu"
        )  # Move to CPU due to sheer big size

        # Softplus function for BPR loss
        self.softplus = nn.Softplus().to(self.device)

    def build_adj_matrix(self, user_ids, item_ids, interaction_scores):
        """
        Build the sparse adjacency matrix from user-item interactions.
        Args:
            user_ids (list or tensor): User indices for each interaction
            item_ids (list or tensor): Item indices for each interaction
            interaction_scores (list or tensor, optional): Interaction scores (optional, binary if None)
        Returns:
            adj_matrix (torch.sparse.FloatTensor): Symmetrically normalized sparse adjacency matrix
        """
        if interaction_scores is None:
            interaction_scores = torch.ones(
                len(user_ids)
            )  # Default to binary interactions if no scores are provided

        # Number of total nodes (users + items)
        n_total_nodes = self.n_users + self.n_items

        # Prepare the adjacency matrix in coordinate (COO) format
        user_tensor = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        item_tensor = (
            torch.tensor(item_ids, dtype=torch.long, device=self.device) + self.n_users
        )  # Shift item indices by n_users
        score_tensor = torch.tensor(
            interaction_scores, dtype=torch.float32, device=self.device
        )

        # Create user-item interaction edges (user-to-item and item-to-user)
        indices = torch.cat([user_tensor.unsqueeze(0), item_tensor.unsqueeze(0)], dim=0)
        values = score_tensor

        # Create the sparse user-item adjacency matrix
        adj_matrix = torch.sparse_coo_tensor(
            indices, values, (n_total_nodes, n_total_nodes)
        ).to(self.device)

        # Symmetric normalization
        row_sum = torch.sparse.sum(
            adj_matrix, dim=1
        ).to_dense()  # Compute row sums (degree for each node)
        d_inv_sqrt = torch.pow(row_sum, -0.5)  # Compute D^-0.5
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = (
            0  # Handle divide by zero for isolated nodes
        )

        # Instead of creating the full diagonal matrix, we scale the rows and columns directly.
        d_inv_sqrt = d_inv_sqrt.to(self.device)

        # Normalize the adjacency matrix using element-wise multiplication
        adj_matrix = adj_matrix.coalesce()  # Ensure it is in coalesced form
        adj_indices = adj_matrix.indices()
        adj_values = adj_matrix.values()

        row_norm = d_inv_sqrt[adj_indices[0]]  # Scale rows
        col_norm = d_inv_sqrt[adj_indices[1]]  # Scale columns

        norm_values = row_norm * adj_values * col_norm  # Element-wise normalization

        # Create normalized adjacency matrix
        adj_matrix = torch.sparse_coo_tensor(
            adj_indices, norm_values, adj_matrix.shape
        ).to(self.device)

        return adj_matrix

    def propagate(self):
        """
        Perform embedding propagation based on the LightGCN propagation rule.
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        all_embeddings = torch.cat(
            [user_embeddings, item_embeddings], dim=0
        )  # Stack user and item embeddings

        # Temporary move to CPU due to big size
        if self.device != "cpu":
            all_embeddings = all_embeddings.to("cpu")

        all_layer_embeddings = [all_embeddings]

        # Perform K-layer propagation
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
            all_embeddings = self.dropout(all_embeddings)
            all_layer_embeddings.append(all_embeddings)

        # Combine embeddings from all layers (mean aggregation)
        final_embeddings = torch.stack(all_layer_embeddings, dim=1).mean(dim=1)

        # Move back the output to device
        if self.device != "cpu":
            final_embeddings = final_embeddings.to(self.device)

        final_user_embeddings = final_embeddings[: self.n_users]  # First part is users
        final_item_embeddings = final_embeddings[self.n_users :]  # Second part is items

        return final_user_embeddings, final_item_embeddings

    def forward(self, users, items):
        """
        Perform forward pass to get final user and item embeddings.
        Args:
            users (tensor): User indices
            items (tensor): Item indices
        """
        user_embeddings, item_embeddings = self.propagate()

        # Get specific user and item embeddings
        return user_embeddings[users], item_embeddings[items]

    def predict(self, users, items):
        """
        Predict interaction score for given users and items.
        """
        user_emb, item_emb = self.forward(users, items)
        return torch.sum(user_emb * item_emb, dim=1)
