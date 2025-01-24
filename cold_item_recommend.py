from parser import parse_args
from ast import parse
import os
import time
import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy.sparse as sp

from models.gaussian_diffusion import GaussianDiffusion
from models.MLP import MLP, ResidualMLP
from modules.dataloader import DataLoaderBuilder
# from modules.trainer_batch_wise import Trainer
from modules.trainer import Trainer
from modules import evaluate_utils

from tqdm import tqdm

import random

def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model, diffusion, args, state='best'):
    # Load saved model weights
    num_layer = len(eval(args.in_dims)) - 1
    model_checkpoint = os.path.join(args.save_path, f'{state}_{args.objective}_{args.noise_schedule}_{num_layer}layer_dropout{args.dropout}_{args.mlp_act_func}_{args.timesteps}timesteps.pt')
    if os.path.exists(model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])
        print("Model loaded successfully from", model_checkpoint)
    else:
        raise FileNotFoundError(f"No model found at {model_checkpoint}")

    return model, diffusion


if __name__ == '__main__':
    set_random_seed(seed=1)
    args = parse_args()
    print("args:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    states = ['best', 'last']
    
    for state in states:
        # Load data and prepare DataLoader
        data_loader_builder = DataLoaderBuilder(args.emb_path, args.tag_emb_path, args.batch_size)
        sample_items, sample_tags, zero_rows = data_loader_builder.load_sample_data()
        bpr_items, bpr_tags, zero_rows = data_loader_builder.load_sample_data(is_cold=False)

        sample_dataloader = data_loader_builder.prepare_dataloaders_sample(sample_items, sample_tags)
        bpr_dataloader = data_loader_builder.prepare_dataloaders_sample(bpr_items, bpr_tags)

        ### model ###
        if args.model == 'MLP':
            model = MLP(
                        in_dims=eval(args.in_dims),
                        time_emb_dim=args.time_emb_dim,
                        tag_emb_dim=args.tag_emb_dim,
                        act_func=args.mlp_act_func,
                        dropout=args.dropout
                        ).cuda()
        elif args.model == 'ResidualMLP':
            model = ResidualMLP(
                                in_dims=eval(args.in_dims),
                                time_emb_dim=args.time_emb_dim,
                                tag_emb_dim=args.tag_emb_dim,
                                act_func=args.mlp_act_func,
                                dropout=args.dropout
                                ).cuda()
        
        diffusion = GaussianDiffusion(
                                    model,
                                    x_size = eval(args.in_dims)[0],
                                    timesteps = args.timesteps,
                                    objective=args.objective,
                                    beta_schedule=args.noise_schedule
                                    ).cuda()
        
        model, diffusion = load_model(model, diffusion, args, state)

        # if args.optimizer == 'Adagrad':
        #     optimizer = optim.Adagrad(
        #         model.parameters(), lr=args.lr, initial_accumulator_value=1e-8, weight_decay=args.wd)
        # elif args.optimizer == 'Adam':
        #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # elif args.optimizer == 'AdamW':
        #     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # elif args.optimizer == 'SGD':
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # elif args.optimizer == 'Momentum':
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=args.wd)

        print("models ready.")

        # Train and validate
        trainer = Trainer(model, diffusion, device, args.num_t_samples, args)
        
        # user
        users = np.load(args.user_path)
        # items (orgin, sample, orgin + sample)
        items_orgin = np.load(args.emb_path).astype(np.float32)
        zero_rows = np.all(items_orgin == 0, axis=1)
        vali_rows = pd.read_csv('./data/ML25M/BPR_cv/cold_movies_vali_0.tsv', sep='\t')['mid'].tolist()
        test_rows = pd.read_csv('./data/ML25M/BPR_cv/cold_movies_test_0.tsv', sep='\t')['mid'].tolist()
        
        # items_bpr : train, val, test of BPR model(non-zero)
        items_bpr, _, zero_idxs = data_loader_builder.load_sample_data(is_cold=False)
        items_sampled = trainer.sample_item_emb(sample_dataloader)
        items_train_sampled = trainer.sample_item_emb(bpr_dataloader) # train dataset (non-zero)
        
        items_all = items_orgin.copy()
        
        items_all[zero_rows] = items_sampled

        ### get average embeddings size ###
        bpr_norms = np.linalg.norm(items_bpr, axis=1)
        average_bpr_norm = np.mean(bpr_norms)
        bpr_mean, bpr_std = evaluate_utils.get_distribution(items_bpr)

        sample_norms = np.linalg.norm(items_train_sampled, axis=1)    
        average_sample_norm = np.mean(sample_norms)
        sample_mean, sample_std = evaluate_utils.get_distribution(items_train_sampled)

        print(f'Avg CF(ground truth) Norm: {average_bpr_norm}')
        print(f'Avg Sampled Norm: {average_sample_norm}')
        evaluate_utils.visualize_distribution(items_bpr, items_train_sampled, count=None, method='heatmap')        
        # print('#'*20)
        # print(f'Mean of each feature orginal ICF: {bpr_mean}')
        # print(f'Std of each feature orginal ICF: {bpr_std}')
        # print('#'*20)
        # print(f'Mean of each feature Sampled ICF: {sample_mean}')
        # print(f'Std of each feature Sampled ICF: {sample_std}')

        # ### sample scaling ###
        # sampled_norms = np.linalg.norm(items_sampled, axis=1)

        # # Calculate the average L2 norm for both original and sampled embeddings
        # average_train_norm = np.mean(train_norms)
        # average_sampled_norm = np.mean(sampled_norms)

        # # Compute the scaling factor to match the norms
        # scaling_factor = average_train_norm / average_sampled_norm

        # # Apply the scaling factor to the sampled embeddings
        # items_sampled = items_sampled * scaling_factor

        # # Apply the scaled sampled embeddings to the zero rows in items_all
        # items_all[zero_rows] = items_sampled

        
        max_k = eval(args.topN)[-1]


        print("#" * 16)
        print('Sample')
        print("#" * 16)
        users = np.load(args.user_path)
        gt_indices = evaluate_utils.get_ground_truth(args.gt_path)

        # 상호작용을 안한 유저, 즉 gt가 없는 유저를 제거
        if abs(len(users) - len(gt_indices)) > 0:
            ratings = pd.read_csv(args.gt_path, sep='\t')
            uids = set(ratings['uid'].unique())

            null_mask = ~np.isin(np.arange(len(users)), list(uids))
            # remove null users
            users = np.delete(users, np.where(null_mask)[0], axis=0)

        # predicted indices
        pred_indices, pred_scores = evaluate_utils.recommend(users, items_sampled, max_k)

        item_idxs = np.where(zero_rows == True)
        pred_indices = item_idxs[0][pred_indices]

        # precision, recall, NDCG, MRR
        pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
        evaluate_utils.print_results(test_result=pred_result, state=state, args=args)


        # print("#" * 16)
        # print('Test(BPR) + Cold-Item')
        # print("#" * 16)
        
        # users = np.load(args.user_path)
        # gt_indices = evaluate_utils.get_ground_truth('data\ML25M\BPR_cv\BPR_test_cold_all_0.tsv')
        # # 상호작용을 안한 유저, 즉 gt가 없는 유저를 제거
        # if abs(len(users) - len(gt_indices)) > 0:
        #     ratings = pd.read_csv('data\ML25M\BPR_cv\BPR_test_cold_all_0.tsv', sep='\t')
        #     uids = set(ratings['uid'].unique())

        #     null_mask = ~np.isin(np.arange(len(users)), list(uids))
        #     # remove null users
        #     users = np.delete(users, np.where(null_mask)[0], axis=0)

        # # predicted indices
        # pred_indices, pred_scores = evaluate_utils.recommend(users, items_orgin, max_k)

        # # precision, recall, NDCG, MRR
        # pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
        # evaluate_utils.print_results(pred_result)
        

        # print("#" * 16)
        # print('Test(BPR) + Cold-Item(Sampled)')
        # print("#" * 16)
        # users = np.load(args.user_path)
        # gt_indices = evaluate_utils.get_ground_truth('data\ML25M\BPR_cv\BPR_test_cold_all_0.tsv')
        # # 상호작용을 안한 유저, 즉 gt가 없는 유저를 제거
        # if abs(len(users) - len(gt_indices)) > 0:
        #     ratings = pd.read_csv('data\ML25M\BPR_cv\BPR_test_cold_all_0.tsv', sep='\t')
        #     uids = set(ratings['uid'].unique())

        #     null_mask = ~np.isin(np.arange(len(users)), list(uids))
        #     # remove null users
        #     users = np.delete(users, np.where(null_mask)[0], axis=0)

        # # predicted indices
        # pred_indices, pred_scores = evaluate_utils.recommend(users, items_all, max_k)

        # # precision, recall, NDCG, MRR
        # pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
        # evaluate_utils.print_results(pred_result)

    
    
