import numpy as np
import pandas as pd
import bottleneck as bn
import torch
import math
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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


def print_results(valid_result=None, test_result=None, loss=None, state=None, args=None):
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
                f"[Valid] {time} \n"
                f"{args}\n"
                f"Prec:\t{valid_precision}\n"
                f"Reca:\t{valid_recall}\n"
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
                f"{args}\n"
                f"Prec:\t{test_precision}\n"
                f"Reca:\t{test_recall}\n"
                f"NDCG:\t{test_ndcg}\n"
                f"MRR:\t{test_mrr}\n"
                f"########################################################\n"
            )
            print(output.strip())
            f.write(output)
            
            if state == 'last':
                f.write(' \n')

        

def get_ground_truth(path):
    gt_data = pd.read_csv(path, sep='\t')
    gt_data = gt_data.sort_values(['uid','rating','mid'], ascending=[True, False, True])
    
    mid_group = gt_data.groupby('uid')['mid'].apply(list).reset_index(drop=True)
    gt_list = mid_group.values.tolist()

    return gt_list


def get_distribution(data, text=None):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if text != None:
        print(f"{text} - Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std


def visualize_distribution(items_bpr, items_sampled, count=None, method='scatter'):
    """
    t-SNE를 사용하여 items_bpr와 items_sampled의 분포를 시각화.

    Parameters:
        items_bpr (numpy.ndarray): BPR 아이템 임베딩
        items_sampled (numpy.ndarray): 샘플링된 아이템 임베딩
        count (int): 샘플링할 데이터 수 (None이면 전체 사용)
        method (str): 'scatter' 또는 'heatmap'으로 시각화 방법 선택
    """
    num_bpr = len(items_bpr)
    num_sampled = len(items_sampled)
    if count is not None and count > 0:
        # 데이터 샘플링 (큰 데이터셋의 계산 효율을 위해)
        num_bpr = min(count, len(items_bpr))
        num_sampled = min(count, len(items_sampled))

    bpr_sample = items_bpr[np.random.choice(len(items_bpr), num_bpr, replace=False)]
    sampled_sample = items_sampled[np.random.choice(len(items_sampled), num_sampled, replace=False)]

    # t-SNE를 사용하여 2D로 차원 축소
    all_data = np.vstack([bpr_sample, sampled_sample])
    tsne = TSNE(n_components=2, random_state=1, perplexity=30, n_iter=300)
    reduced_data = tsne.fit_transform(all_data)

    # 분리
    total_bpr_sample = len(bpr_sample)
    bpr_tsne = reduced_data[:total_bpr_sample]
    sampled_tsne = reduced_data[total_bpr_sample:]
    
    # x축과 y축의 범위 설정
    x_min = min(bpr_tsne[:, 0].min(), sampled_tsne[:, 0].min())
    x_max = max(bpr_tsne[:, 0].max(), sampled_tsne[:, 0].max())
    y_min = min(bpr_tsne[:, 1].min(), sampled_tsne[:, 1].min())
    y_max = max(bpr_tsne[:, 1].max(), sampled_tsne[:, 1].max())

    if method == 'scatter':
        # 산점도 방식
        plt.figure(figsize=(8, 8))
        plt.scatter(bpr_tsne[:, 0], bpr_tsne[:, 1], label='BPR Items', alpha=0.6, s=15, c='blue')
        plt.scatter(sampled_tsne[:, 0], sampled_tsne[:, 1], label='Sampled Items', alpha=0.6, s=15, c='orange')
        plt.title("t-SNE Scatter Visualization of BPR and Sampled Items")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid(True)
        plt.show()

    elif method == 'heatmap':
        # 원형 히트맵 방식
        plt.figure(figsize=(12, 6))

        # BPR Items 원형 히트맵
        plt.subplot(1, 2, 1)
        bpr_density = gaussian_kde(bpr_tsne.T)(bpr_tsne.T)
        plt.scatter(bpr_tsne[:, 0], bpr_tsne[:, 1], c=bpr_density, cmap='Blues', s=20, alpha=0.7)
        plt.colorbar(label="Density")
        plt.title("Circular Heatmap of BPR Items")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Sampled Items 원형 히트맵
        plt.subplot(1, 2, 2)
        sampled_density = gaussian_kde(sampled_tsne.T)(sampled_tsne.T)
        plt.scatter(sampled_tsne[:, 0], sampled_tsne[:, 1], c=sampled_density, cmap='Oranges', s=20, alpha=0.7)
        plt.colorbar(label="Density")
        plt.title("Circular Heatmap of Sampled Items")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Invalid method. Choose 'scatter' or 'heatmap'.")