from parser import parse_args
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
from models.MLP import MLP, ResidualMLP
from modules.dataloader import DataLoaderBuilder
# from modules.trainer_batch_wise import Trainer
from modules.trainer import Trainer
from modules.evaluate_utils import get_distribution
from tqdm import tqdm

import random

def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_random_seed(seed=1)
    args = parse_args()
    print("args:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Load data and prepare DataLoader
    data_loader_builder = DataLoaderBuilder(args.emb_path, args.tag_emb_path, args.batch_size)
    # remove cold item and split dataset(ICF emb, tag emb) to train, valid and test
    train_items, valid_items, test_items, train_tags, valid_tags, test_tags = data_loader_builder.load_data()
    train_loader, valid_loader, test_loader = data_loader_builder.prepare_dataloaders(
                                                                                      train_items, valid_items, test_items, 
                                                                                      train_tags, valid_tags, test_tags
                                                                                      )
    # bound for data clipping
    mean, std = get_distribution(train_items)
    lower_bound = None
    upper_bound = None
    if args.clamp_k != None:
        lower_bound = torch.tensor(mean - args.clamp_k * std, dtype=torch.float32).cuda()
        upper_bound = torch.tensor(mean + args.clamp_k * std, dtype=torch.float32).cuda()

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
                                  beta_schedule=args.noise_schedule,
                                  lower_bound=lower_bound,
                                  upper_bound=upper_bound
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