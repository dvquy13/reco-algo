import numpy as np
from scipy.sparse import issparse


def chunk_transform(df, pipeline, chunk_size=1000, progress_bar_type="tqdm"):
    if progress_bar_type == "tqdm_notebook":
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    transformed_chunks = []

    progress_bar = tqdm(range(0, df.shape[0], chunk_size), desc="Transforming chunks")

    # Iterate through the DataFrame in chunks
    for start in progress_bar:
        end = min(start + chunk_size, df.shape[0])
        chunk_df = df.iloc[start:end]

        # Apply the pipeline transformation to the chunk
        transformed_chunk = pipeline.transform(chunk_df)

        # Check if the transformed output is sparse, and convert to dense
        if issparse(transformed_chunk):
            transformed_chunk = transformed_chunk.toarray()

        # Collect the transformed chunk
        transformed_chunks.append(transformed_chunk)

    # Concatenate the transformed chunks into a single NumPy array
    transformed_full = np.vstack(transformed_chunks)

    return transformed_full
