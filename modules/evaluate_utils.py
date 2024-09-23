import numpy as np
import bottleneck as bn
import torch
import math


def recommend(user_embeddings, item_embeddings):
    """
    Recommends the top k items for each user in a batch based on user and item embeddings.
    
    Parameters:
    - user_embeddings: np.array of shape (batch_size, latent_dim), the embeddings of the users.
    - item_embeddings: np.array of shape (num_items, latent_dim), the embeddings of the items.
    - k: int, the number of top items to recommend for each user.

    Returns:
    - top_k_indices: np.array of shape (batch_size, k), the indices of the top k recommended items for each user.
    - top_k_scores: np.array of shape (batch_size, k), the scores of the top k recommended items for each user.
    """
    # Compute the dot product between each user embedding and each item embedding
    # This will result in a score matrix of shape (batch_size, num_items)
    scores = np.dot(user_embeddings, item_embeddings.T)

    # Get the indices of the top k items for each user
    indices = np.argsort(scores, axis=1)[:, ::-1][:, :]

    # Get the top k scores for each user
    batch_indices = np.arange(user_embeddings.shape[0])[:, None]  # Shape: (batch_size, 1)
    top_k_scores = scores[batch_indices, indices]

    return indices, top_k_scores


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR


def print_results(valid_result, test_result=None, loss=None):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        

def get_ground_truth(path):
    gt_data = np.load(path)
    
    mid_group = gt_data.groupby('uid')['mid'].apply(list).reset_index()
    gt_list = mid_group.values.tolist()

    return gt_list
