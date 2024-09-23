import argparse
from ast import parse
import os
import time
import numpy as np
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

    # Data paths
    parser.add_argument('--emb_path', type=str, default='./data/ML25M/BPR_cv/BPR_ivec_0.npy', help='load emb path')
    parser.add_argument('--user_path', type=str, default='./data/ML25M/BPR_cv/BPR_uvec_0.npy', help='load user emb path')
    parser.add_argument('--tag_emb_path', type=str, default='./data/ML25M/mv-tag-emb.npy', help='load tag emb path')
    parser.add_argument('--model_path', type=str, default='./saved_models', help='model path to save')
    parser.add_argument('--valid_path', type=str, default='./data/ML25M/BPR_cv/cold_movies_rating_vali_0.tsv', help='preference items of each user')
    parser.add_argument('--test_path', type=str, default='./data/ML25M/BPR_cv/cold_movies_rating_test_0.tsv', help='preference items of each user')

    # Model and training parameters
    parser.add_argument('--num_t_samples', type=int, default=1, help='number of time(t) samples for training') ###
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for MLP')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay for MLP')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
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
    parser.add_argument('--objective', type=str, default='pred_noise', help='objective type: pred_noise, pred_x0, pred_v')
    parser.add_argument('--timesteps', type=int, default=1000, help='diffusion steps') ###
    parser.add_argument('--noise_schedule', type=str, default='sigmoid', help='the schedule for noise generating')
    
    return parser.parse_args()


def load_model(model, diffusion, args, device):
    # Load saved model weights
    model_checkpoint = os.path.join(args.model_path, 'diffusion_model.pth')
    if os.path.exists(model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint))
        print("Model loaded successfully from", model_checkpoint)
    else:
        raise FileNotFoundError(f"No model found at {model_checkpoint}")

    return model, diffusion


def recommend_top_k_items_batch(user_embeddings, item_embeddings, k):
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
    top_k_indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]

    # Get the top k scores for each user
    batch_indices = np.arange(user_embeddings.shape[0])[:, None]  # Shape: (batch_size, 1)
    top_k_scores = scores[batch_indices, top_k_indices]

    return top_k_indices, top_k_scores


if __name__ == '__main__':
    set_random_seed(seed=1)
    args = parse_args()
    print("args:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Load data and prepare DataLoader
    data_loader_builder = DataLoaderBuilder(args.emb_path, args.user_path, args.tag_emb_path, args.batch_size)
    users, items, tags = data_loader_builder.load_vt_data()
    dataloader = data_loader_builder.prepare_dataloaders(users, items, tags)

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
    users, items = trainer.sample_item_emb(dataloader)


