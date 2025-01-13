import numpy as np
import pandas as pd
import bottleneck as bn
import torch
import math
from datetime import datetime

def recommend(user_embeddings, item_embeddings, max_k):
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
    item_indices = np.argsort(scores, axis=1)[:, ::-1][:, :max_k].tolist()

    # Get the top k scores for each user
    user_indices = np.arange(user_embeddings.shape[0])[:, None]  # Shape: (batch_size, 1)
    top_k_scores = scores[user_indices, item_indices]

    return item_indices, top_k_scores


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
        
        precision.append(round(sumForPrecision / len(predictedIndices), 6))
        recall.append(round(sumForRecall / len(predictedIndices), 6))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 6))
        MRR.append(round(sumForMRR / len(predictedIndices), 6))
        
    return precision, recall, NDCG, MRR


def print_results(valid_result=None, test_result=None, loss=None, state=None):
    """Output the evaluation results with tab-separated format for easy Excel pasting."""
    file_path = './result/evaluation_results.txt'
    with open(file_path, 'a') as f:
        if loss is not None:
            output = f"[Train]\tloss:\t{loss:.4f}\n"
            print(output.strip())
            f.write(output)
        if valid_result is not None:
            valid_precision = '\t'.join([str(x) for x in valid_result[0]])
            valid_recall = '\t'.join([str(x) for x in valid_result[1]])
            valid_ndcg = '\t'.join([str(x) for x in valid_result[2]])
            valid_mrr = '\t'.join([str(x) for x in valid_result[3]])

            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = (
                f"[Valid] {time}\n"
                f"Pr:\t{valid_precision}\n"
                f"Re:\t{valid_recall}\n"
                f"NDCG:\t{valid_ndcg}\n"
                f"MRR:\t{valid_mrr}\n"
                f"########################################################\n"
            )
            print(output.strip())
            f.write(output)
        if test_result is not None:
            test_precision = '\t'.join([str(x) for x in test_result[0]])
            test_recall = '\t'.join([str(x) for x in test_result[1]])
            test_ndcg = '\t'.join([str(x) for x in test_result[2]])
            test_mrr = '\t'.join([str(x) for x in test_result[3]])

            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = (
                f"[Test with {state} model] {time}\n"
                f"Pr:\t{test_precision}\n"
                f"Re:\t{test_recall}\n"
                f"NDCG:\t{test_ndcg}\n"
                f"MRR:\t{test_mrr}\n"
                f"########################################################\n"
            )
            print(output.strip())
            f.write(output)

        

def get_ground_truth(path):
    gt_data = pd.read_csv(path, sep='\t')
    gt_data = gt_data.sort_values(['uid','rating','mid'], ascending=[True, False, True])
    
    mid_group = gt_data.groupby('uid')['mid'].apply(list).reset_index(drop=True)
    gt_list = mid_group.values.tolist()

    return gt_list
