import numpy as np

def recommend_top_k_items(user_embedding, item_embeddings, k=10):
    """
    Recommends the top k items based on user and item embeddings.
    
    Parameters:
    - user_embedding: np.array of shape (latent_dim,), the embedding of the user.
    - item_embeddings: np.array of shape (num_items, latent_dim), the embeddings of the items.
    - k: int, the number of top items to recommend.

    Returns:
    - top_k_indices: List of indices of the top k recommended items.
    - top_k_scores: List of scores of the top k recommended items.
    """
    # Compute the dot product between the user embedding and each item embedding
    scores = np.dot(item_embeddings, user_embedding)
    
    # Get the indices of the top k items based on the scores
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    # Get the scores of the top k items
    top_k_scores = scores[top_k_indices]
    
    return top_k_indices, top_k_scores


# Example usage
if __name__ == "__main__":
    # Random example data
    user_emb_path = 'data\ML25M\BPR_cv\BPR_uvec_0.npy'
    item_emb_path = 'data\ML25M\BPR_cv\BPR_ivec_0.npy'

    user_embedding = np.load(user_emb_path)
    item_embeddings = np.load(item_emb_path)

    val_test_idx = np.all(item_embeddings == 0, axis=1)
    if np.any(val_test_idx):
        print(np.where(val_test_idx == True))
        print(f"There are {np.sum(val_test_idx)} valid&test items with zero embedding.")
    else:
        print("There are no valid&test embedding.")

    val_user_emb = user_embedding[val_test_idx]
    val_item_emb = item_embeddings[val_test_idx]

    k = 5  # Number of items to recommend
    top_k_indices, top_k_scores = recommend_top_k_items(user_embedding, item_embeddings, k)

    print("Top K Recommended Item Indices:", top_k_indices)
    print("Top K Recommended Item Scores:", top_k_scores)