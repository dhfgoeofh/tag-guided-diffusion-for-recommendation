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
from models.MLP_wo_tag_encode import MLP

from tqdm import tqdm

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--emb_path', type=str, default='./data/ML25M/BPR_cv/BPR_ivec_0.npy', help='load emb path')
parser.add_argument('--tag_emb_path', type=str, default='./data/ML25M/mv-tag-emb.npy', help='load tag emb path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for MLP')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay for MLP')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1500, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the MLP
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--num_layers', type=int, default=1, help='number of MLP layers')
parser.add_argument('--in_dims', type=str, default='[128]', help='the dims for item embedding')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--tag_emb_dim', type=int, default='400', help='the dims for tag embedding')
parser.add_argument('--time_emb_dim', type=int, default=10, help='timestep embedding size')
parser.add_argument('--mlp_act_func', type=str, default='tanh', help='the activation function for MLP')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for MLP: Adam, AdamW, SGD, Adagrad, Momentum')

# params for diffusion
parser.add_argument('--objective', type=str, default='pred_noise', help='objective type: pred_noise, pred_x0, pred_v')
parser.add_argument('--steps', type=int, default=1000, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001)
parser.add_argument('--noise_max', type=float, default=0.02)
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=10, help='steps for sampling/denoising')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
# parser.add_argument('--num_classes', type=int, default=10, help='condition classes for classifier')

if __name__ == '__main__':
    args = parser.parse_args()
    print("args:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ### DATA LOAD ###
    item_embeddings = np.load(args.emb_path).astype(np.float32)  # Replace with actual item embeddings (13816, 128)
    tag_embeddings = np.load(args.tag_emb_path).astype(np.float32)   # Replace with actual tag embeddings (13816, 400)
    
    # find zero rows
    zero_rows = np.all(item_embeddings == 0, axis=1)

    # remove zero data
    item_embeddings_cleaned = item_embeddings[~zero_rows]
    tag_embeddings_cleaned = tag_embeddings[~zero_rows]

    print(f'Origin item data: {item_embeddings.shape}, Cleaned item data: {item_embeddings_cleaned.shape}')
    print(f'Origin tag data: {tag_embeddings.shape}, Cleaned tag data: {tag_embeddings_cleaned.shape}')

    # Split the data into train, validation, and test sets (80% train, 10% val, 10% test)
    train_items, temp_items, train_tags, temp_tags = train_test_split(item_embeddings_cleaned, tag_embeddings_cleaned, test_size=0.2, random_state=42)
    valid_items, test_items, valid_tags, test_tags = train_test_split(temp_items, temp_tags, test_size=0.5, random_state=42)
    batch_size = args.batch_size
    # print(train_items[0], train_tags[0])

    train_items = torch.tensor(train_items, dtype=torch.float32)
    train_tags = torch.tensor(train_tags, dtype=torch.float32)
    valid_items = torch.tensor(valid_items, dtype=torch.float32)
    valid_tags = torch.tensor(valid_tags, dtype=torch.float32)
    test_items = torch.tensor(test_items, dtype=torch.float32)
    test_tags = torch.tensor(test_tags, dtype=torch.float32)

    train_dataset = TensorDataset(train_items, train_tags)
    valid_dataset = TensorDataset(valid_items, valid_tags)
    test_dataset = TensorDataset(test_items, test_tags)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for layer_num in [1, 2, 3, 4]:
        ### model ###
        # num_classes = 10
        model = MLP(
                    in_dims=eval(args.in_dims),
                    out_dims=eval(args.in_dims),
                    time_emb_dim=args.time_emb_dim,
                    tag_emb_dim=args.tag_emb_dim,
                    act_func=args.mlp_act_func,
                    num_layers=layer_num
                    ).cuda()
        print("Noise Scheduler: ",args.noise_schedule)
        
        diffusion = GaussianDiffusion(
                                    model,
                                    x_size = 128,
                                    timesteps = 1000,
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

        ## Initialize best validation loss tracker
        best_valid_loss = float('inf')
        best_epoch = -1

        ## training ##
        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                item_batch, tag_batch = batch
                item_batch, tag_batch = item_batch.cuda(), tag_batch.cuda()
                        
                # Forward pass through the diffusion process
                # print("Item Batch Shape:",item_batch.shape)
                loss = diffusion(item_batch, classes=tag_batch)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            model.eval()
            total_valid_loss = 0
            
            with torch.no_grad():
                for batch in valid_loader:
                    item_batch, tag_batch = batch
                    item_batch, tag_batch = item_batch.cuda(), tag_batch.cuda()
                    
                    # Forward pass through the diffusion process
                    loss = diffusion(item_batch, classes=tag_batch)
                    total_valid_loss += loss.item()
            
            avg_valid_loss = total_valid_loss / len(valid_loader)

            # Check if current validation loss is the best so far
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_epoch = epoch + 1  # Store best epoch (1-indexed)

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")
        
        ## After training, evaluate on test set ##
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for batch in test_loader:
                item_batch, tag_batch = batch
                item_batch, tag_batch = item_batch.cuda(), tag_batch.cuda()

                # Forward pass through the diffusion process
                loss = diffusion(item_batch, classes=tag_batch)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss}")

        # Print out the best epoch for reference
        print(f"Best Epoch: {best_epoch}, Best Validation Loss: {best_valid_loss}")
        print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
