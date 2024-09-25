from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import Embedding


class TorchEmbeddingStore:
    def __init__(self, id_mapper: Dict, embeddings: Embedding):
        assert "id_to_idx" in id_mapper
        assert "idx_to_id" in id_mapper
        self.id_mapper = id_mapper
        self.embeddings = embeddings
        self.embedding_dim = embeddings(torch.tensor(0)).shape[0]

    def find_single_idx(self, single_inp):
        if isinstance(single_inp, str):
            return self.id_mapper["id_to_idx"][single_inp]
        return single_inp

    def get_emb(self, inp: Union[List, str, int], batch_size=128):
        # If input is a list, process it in batches
        if isinstance(inp, (List, np.ndarray)):
            indices = [self.find_single_idx(item) for item in inp]
            embeddings = []

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_embeddings = self.embeddings(torch.tensor(batch_indices))
                embeddings.append(batch_embeddings)

            return torch.cat(embeddings, dim=0).detach().numpy()

        # If input is a single element, process it directly
        else:
            idx = self.find_single_idx(inp)
            return self.embeddings(torch.tensor(idx))

    def save(self, file_path: str):
        """
        Save the id_mapper and embeddings state to a file.
        """
        torch.save(
            {
                "embedding_weights": self.embeddings.state_dict(),
                "id_mapper": self.id_mapper,
                "num_embeddings": self.embeddings.num_embeddings,
                "embedding_dim": self.embeddings.embedding_dim,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str):
        """
        Load the id_mapper and embeddings state from a file.

        Parameters:
        file_path (str): Path to the file where the store is saved.

        Returns:
        TorchEmbeddingStore: An instance of TorchEmbeddingStore with loaded state.
        """
        # Load the checkpoint
        checkpoint = torch.load(file_path)

        # Retrieve num_embeddings and embedding_dim from the saved checkpoint
        num_embeddings = checkpoint["num_embeddings"]
        embedding_dim = checkpoint["embedding_dim"]

        # Recreate the Embedding layer
        embeddings = Embedding(num_embeddings, embedding_dim)
        embeddings.load_state_dict(checkpoint["embedding_weights"])

        # Return an instance of TorchEmbeddingStore with loaded id_mapper and embeddings
        return cls(id_mapper=checkpoint["id_mapper"], embeddings=embeddings)


class TorchEmbeddingStorePipeline:
    def __init__(self, embs: TorchEmbeddingStore, id_col: str):
        self.id_col = id_col
        self.embs = embs

    def transform(self, df: pd.DataFrame):
        return self.embs.get_emb(df[self.id_col].values)
