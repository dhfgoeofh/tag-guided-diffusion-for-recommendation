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
    parser.add_argument('--tag_emb_path', type=str, default='./data/ML25M/mv-tag-emb.npy', help='load tag emb path')

    # Model and training parameters
    parser.add_argument('--num_t_samples', type=int, default=5, help='number of time(t) samples for training') ###
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
    parser.add_argument('--objective', type=str, default='pred_noise', help='objective type: pred_noise, pred_x0, pred_v')
    parser.add_argument('--timesteps', type=int, default=250, help='diffusion steps') ###
    parser.add_argument('--noise_schedule', type=str, default='linear', help='the schedule for noise generating')
    
    return parser.parse_args()


if __name__ == '__main__':
    set_random_seed(seed=1)
    args = parse_args()
    print("args:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Load data and prepare DataLoader
    data_loader_builder = DataLoaderBuilder(args.emb_path, args.tag_emb_path, args.batch_size)
    train_items, valid_items, test_items, train_tags, valid_tags, test_tags = data_loader_builder.load_data()
    train_loader, valid_loader, test_loader = data_loader_builder.prepare_dataloaders(
                                                                                      train_items, valid_items, test_items, 
                                                                                      train_tags, valid_tags, test_tags
                                                                                      )

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

    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(), lr=args.lr, initial_accumulator_value=1e-8, weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=args.wd)
    print("models ready.")


    # Train and validate
    trainer = Trainer(model, diffusion, device, args.num_t_samples, args)
    trainer.train(train_loader, valid_loader)

    # Test
    trainer.test(test_loader)