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
    user_embedding = np.array([0.2, 0.3, 0.5])  # Example user embedding
    item_embeddings = np.array([[0.1, 0.4, 0.6], 
                                [0.2, 0.1, 0.9], 
                                [0.9, 0.7, 0.3], 
                                [0.4, 0.6, 0.8]])  # Example item embeddings

    k = 2  # Number of items to recommend
    top_k_indices, top_k_scores = recommend_top_k_items(user_embedding, item_embeddings, k)

    print("Top K Recommended Item Indices:", top_k_indices)
    print("Top K Recommended Item Scores:", top_k_scores)