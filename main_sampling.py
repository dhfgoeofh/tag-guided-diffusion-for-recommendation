import argparse
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
from models.MLP import MLP
from modules.dataloader import DataLoaderBuilder
# from modules.trainer_batch_wise import Trainer
from modules.trainer_batch_wise import Trainer
from modules import evaluate_utils

from tqdm import tqdm

import random

def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()

    # Evaluation Parameters
    parser.add_argument('--topN', type=str, default='[5, 10, 30, 50]', help='top N items for evaluation.')
    parser.add_argument('--gt_path', type=str, default='./data/ML25M/BPR_cv/cold_movies_rating_all_0.tsv', help='preference items of each user')

    # Data paths
    parser.add_argument('--emb_path', type=str, default='./data/ML25M/BPR_cv/BPR_ivec_0.npy', help='load emb path')
    parser.add_argument('--user_path', type=str, default='./data/ML25M/BPR_cv/BPR_uvec_0.npy', help='load user emb path')
    parser.add_argument('--tag_emb_path', type=str, default='./data/ML25M/mv-tag-emb.npy', help='load tag emb path')
    
    # Model and training parameters
    parser.add_argument('--num_t_samples', type=int, default=1, help='number of time(t) samples for training') ###
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for MLP')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay for MLP')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')

    # MLP parameters
    parser.add_argument('--num_layers', type=int, default=1, help='number of MLP layers')
    parser.add_argument('--in_dims', type=int, default=128, help='the dims for item embedding')
    parser.add_argument('--tag_emb_dim', type=int, default=400, help='the dims for tag embedding')
    parser.add_argument('--time_emb_dim', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--mlp_act_func', type=str, default='tanh', help='the activation function for MLP')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for MLP: Adam, AdamW, SGD, etc.')

    # Diffusion parameters
    parser.add_argument('--noise_scale', type=float, default=0.005, help='noise scale')
    parser.add_argument('--objective', type=str, default='pred_x0', help='objective type: pred_noise, pred_x0, pred_v')
    parser.add_argument('--timesteps', type=int, default=1500, help='diffusion steps') ###
    parser.add_argument('--noise_schedule', type=str, default='linear', help='the schedule for noise generating')
    
    return parser.parse_args()


def load_model(model, diffusion, args, device):
    # Load saved model weights
    model_checkpoint = os.path.join(args.save_path, f'best_{args.objective}_{args.timesteps}timesteps.pt')
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

    # Load data and prepare DataLoader
    data_loader_builder = DataLoaderBuilder(args.emb_path, args.tag_emb_path, args.batch_size)
    items, tags, zero_rows = data_loader_builder.load_vt_data()
    dataloader = data_loader_builder.prepare_dataloaders_vt(items, tags)

    ### model ###
    model = MLP(
                in_dims=[args.in_dims],
                out_dims=[args.in_dims],
                time_emb_dim=args.time_emb_dim,
                tag_emb_dim=args.tag_emb_dim,
                act_func=args.mlp_act_func,
                num_layers=args.num_layers
                ).cuda()
    
    diffusion = GaussianDiffusion(
                                  model,
                                  x_size = args.in_dims,
                                  timesteps = args.timesteps,
                                  objective=args.objective,
                                  beta_schedule=args.noise_schedule
                                  ).cuda()
    
    model, diffusion = load_model(model, diffusion, args, device)

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
    
    # items_bpr : train, val, test of BPR model(non-zero)
    items_bpr, _, zero_idxs = data_loader_builder.load_vt_data(is_sample=False)
    items_sampled = trainer.sample_item_emb(dataloader)
    items_all = items_orgin.copy()
    
    items_all[zero_rows] = items_sampled

    ### get average embeddings size ###
    bpr_norms = np.linalg.norm(items_bpr, axis=1)
    average_bpr_norm = np.mean(bpr_norms)

    sample_norms = np.linalg.norm(items_sampled, axis=1)    
    average_sample_norm = np.mean(sample_norms)

    print(f'Avg BPR Norm: {average_bpr_norm}')
    print(f'Avg Sampled Norm: {average_sample_norm}')

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
    print('Test(BPR)')
    print("#" * 16)
    users_idx = pd.read_csv('data\ML25M\BPR_cv\BPR_test_0.tsv', sep='\t')['uid'].unique()
    users = np.load(args.user_path)[users_idx]
    
    gt_indices = evaluate_utils.get_ground_truth('data\ML25M\BPR_cv\BPR_test_0.tsv')

    # 상호작용을 안한 유저, 즉 gt가 없는 유저를 제거
    if abs(len(users) - len(gt_indices)) > 0:
        # ratings = pd.read_csv('data\ML25M\BPR_cv\BPR_test_0.tsv', sep='\t')
        # uids = set(ratings['uid'].unique())

        # null_mask = ~np.isin(np.arange(len(users)), list(uids))
        # # remove null users
        # users = np.delete(users, np.where(null_mask)[0], axis=0)
        print('num_user != num_gt_user')
        
    # predicted indices
    pred_indices, pred_scores = evaluate_utils.recommend(users, items_bpr, max_k)

    item_idxs = np.where(zero_rows == False)
    pred_indices = item_idxs[0][pred_indices]
 
    # precision, recall, NDCG, MRR
    pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
    evaluate_utils.print_results(pred_result)


    print("#" * 16)
    print('Random')
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

    ## predicted indices
    # pred_indices, pred_scores = evaluate_utils.recommend(users, items_sampled, max_k)
    item_idxs = np.where(zero_rows == True)

    ## Random prediction
    matrix = np.zeros((len(users), max_k), dtype=int)
    
    for i in range((len(users))):
        matrix[i] = np.random.choice(len(item_idxs[0]), size=max_k, replace=False)

    pred_indices = item_idxs[0][matrix]

    # precision, recall, NDCG, MRR
    pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
    evaluate_utils.print_results(pred_result)


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
    evaluate_utils.print_results(pred_result)


    print("#" * 16)
    print('Non-zero + Zeros')
    print("#" * 16)
    
    users = np.load(args.user_path)
    gt_indices = evaluate_utils.get_ground_truth('data\ML25M\BPR_cv\BPR_all_cold_all_0.tsv')
    # 상호작용을 안한 유저, 즉 gt가 없는 유저를 제거
    if abs(len(users) - len(gt_indices)) > 0:
        ratings = pd.read_csv('data\ML25M\BPR_cv\BPR_all_cold_all_0.tsv', sep='\t')
        uids = set(ratings['uid'].unique())

        null_mask = ~np.isin(np.arange(len(users)), list(uids))
        # remove null users
        users = np.delete(users, np.where(null_mask)[0], axis=0)

    # predicted indices
    pred_indices, pred_scores = evaluate_utils.recommend(users, items_orgin, max_k)

    # precision, recall, NDCG, MRR
    pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
    evaluate_utils.print_results(pred_result)
    

    print("#" * 16)
    print('Non-zero + Sample')
    print("#" * 16)
    users = np.load(args.user_path)
    gt_indices = evaluate_utils.get_ground_truth('data\ML25M\BPR_cv\BPR_all_cold_all_0.tsv')
    # 상호작용을 안한 유저, 즉 gt가 없는 유저를 제거
    if abs(len(users) - len(gt_indices)) > 0:
        ratings = pd.read_csv('data\ML25M\BPR_cv\BPR_all_cold_all_0.tsv', sep='\t')
        uids = set(ratings['uid'].unique())

        null_mask = ~np.isin(np.arange(len(users)), list(uids))
        # remove null users
        users = np.delete(users, np.where(null_mask)[0], axis=0)

    # predicted indices
    pred_indices, pred_scores = evaluate_utils.recommend(users, items_all, max_k)

    # precision, recall, NDCG, MRR
    pred_result = evaluate_utils.computeTopNAccuracy(gt_indices, pred_indices, eval(args.topN))
    evaluate_utils.print_results(pred_result)

    
    
