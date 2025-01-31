import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing as prep
from scipy.sparse import lil_matrix, csr_matrix, load_npz
from tqdm import tqdm
import argparse
import time

# --- Timer for logging ---
class Timer:
    def __init__(self, name='default'):
        self.name = name
        self.start_time = time.time()

    def tic(self):
        self.start_time = time.time()

    def toc(self, message=''):
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] {message} elapsed: {elapsed:.2f}s")



# --- Helper functions ---
def l2_norm(tensor):
    return torch.sum(tensor ** 2)

def tfidf(x):
    """
    Compute TF-IDF transformation on a sparse matrix x.
    """
    if not isinstance(x, csr_matrix):
        x = csr_matrix(x)

    # Term Frequency (TF)
    x_tf = x.copy()
    x_tf.data = np.log(x_tf.data + 1)

    # Inverse Document Frequency (IDF)
    num_docs = x.shape[0]
    doc_freq = np.asarray((x > 0).sum(axis=0)).flatten()  # Document frequency for each term
    idf = np.log((num_docs - 1) / (doc_freq + 1))

    # Apply IDF scaling to each term
    x_tfidf = x_tf.multiply(idf)

    return x_tfidf

def standardize(x):
    scaler = prep.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_scaled = np.clip(x_scaled, -5, 5)
    return scaler, x_scaled

def negative_sampling(user_list, item_list, neg, item_warm):
    neg_samples = int(neg)
    user_pos = user_list.reshape((-1))
    user_neg = np.tile(user_list, neg_samples).reshape((-1))
    pos_items = item_list.reshape((-1))
    neg_items = np.random.choice(item_warm, size=(neg_samples * user_list.shape[0]), replace=True).reshape((-1))
    target_pos = np.ones_like(pos_items)
    target_neg = np.zeros_like(neg_items)
    return np.concatenate((user_pos, user_neg)), np.concatenate((pos_items, neg_items)), np.concatenate((target_pos, target_neg))



def load_data(data_name, dataset='CiteULike'):
    timer = Timer(name='Data Loading')
    timer.tic()

    data_path = f'./data/{data_name}'
    u_pref = np.load(f'{data_path}/U_BPR.npy')
    v_pref = np.load(f'{data_path}/V_BPR.npy')

    item_content = None
    user_content = None
    if dataset == 'CiteULike':
        item_content_file = f'{data_path}/item_features.txt'
        item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
        item_content = tfidf(item_content)
    elif dataset == 'LastFM':
        user_content_file = f'{data_path}/user_content.npz'
        user_content = load_npz(user_content_file)
        user_content = user_content.tolil(copy=False)
    elif dataset == 'XING':
        user_content_file = f'{data_path}/user_content.npz'
        user_content = load_npz(user_content_file)
        user_content = user_content.tolil(copy=False)

        item_content_file = f'{data_path}/item_content.npz'
        item_content = load_npz(item_content_file)
        item_content = item_content.tolil(copy=False)
    else:
        print("Cant load Contents data. Check the dataset.")

    from sklearn.utils.extmath import randomized_svd
    u, s, _ = randomized_svd(item_content, n_components=300, n_iter=5)
    item_content = u * s

    _, item_content = standardize(item_content)
    _, u_pref = standardize(u_pref)
    _, v_pref = standardize(v_pref)

    train = pd.read_csv(f'{data_path}/train.csv', dtype=np.int32)
    target = np.ones(len(train))

    data = {
        'u_pref': u_pref,
        'v_pref': v_pref,
        'u_content': user_content,
        'v_content': item_content,
        'user_list': train['uid'].values,
        'item_list': train['iid'].values,
        'target': target
    }

    timer.toc('Data loaded successfully.')
    return data



# --- Model definition ---
class Heater(nn.Module):
    def __init__(self, latent_dim, content_dim, output_dim, num_experts=5, random_prob=0.5, dropout=0.5, alpha=0.1, beta=0.01):
        super(Heater, self).__init__()
        self.num_experts = num_experts
        self.random_prob = random_prob
        self.dropout = dropout
        self.alpha = alpha  # 차이 손실 가중치
        self.beta = beta    # 정규화 손실 가중치

        # Mixture of Experts for content embedding transformation
        self.experts_user = nn.ModuleList([nn.Linear(content_dim, output_dim) for _ in range(num_experts)])
        self.experts_item = nn.ModuleList([nn.Linear(content_dim, output_dim) for _ in range(num_experts)])
        self.gate_user = nn.Linear(content_dim, num_experts)
        self.gate_item = nn.Linear(content_dim, num_experts)

        # CF embedding layers
        self.user_cf_layer = nn.Linear(latent_dim, output_dim)
        self.item_cf_layer = nn.Linear(latent_dim, output_dim)

    def transform_content(self, content, gate_layer, experts):
        gate_values = torch.softmax(gate_layer(content), dim=1)
        expert_outputs = torch.stack([expert(content) for expert in experts], dim=1)
        output = torch.sum(gate_values.unsqueeze(2) * expert_outputs, dim=1)
        return output

    def calculate_reg_loss(self):
        # 정규화 손실(L2 정규화)
        reg_loss = sum(l2_norm(param) for param in self.parameters()) * self.beta
        return reg_loss

    def calculate_diff_loss(self, u_content_transformed, v_content_transformed, u_pref_cf, v_pref_cf):
        # 차이 손실 (CF 임베딩과 콘텐츠 임베딩 간의 차이 계산)
        diff_user_loss = l2_norm(u_content_transformed - u_pref_cf) if u_content_transformed is not None else 0
        diff_item_loss = l2_norm(v_content_transformed - v_pref_cf) if v_content_transformed is not None else 0
        return self.alpha * (diff_user_loss + diff_item_loss)

    def forward(self, u_pref, u_content, v_pref, v_content, dataset='CiteULike'):
        # Apply Mixture of Experts to transform content embeddings
        u_content_transformed, v_content_transformed = None, None

        if dataset == 'CiteULike':
            v_content_transformed = self.transform_content(v_content, self.gate_item, self.experts_item)
        elif dataset == 'LastFM':
            u_content_transformed = self.transform_content(u_content, self.gate_user, self.experts_user)
        elif dataset == 'XING':
            u_content_transformed = self.transform_content(u_content, self.gate_user, self.experts_user)
            v_content_transformed = self.transform_content(v_content, self.gate_item, self.experts_item)

        # Get CF embeddings
        u_pref_cf = self.user_cf_layer(u_pref)
        v_pref_cf = self.item_cf_layer(v_pref)

        # Randomized Training: Choose between CF or content embeddings
        if np.random.rand() < self.random_prob:
            user_input = u_pref_cf
            item_input = v_pref_cf
        else:
            user_input = u_content_transformed if u_content_transformed is not None else u_pref_cf
            item_input = v_content_transformed if v_content_transformed is not None else v_pref_cf

        # Compute output prediction
        output = (user_input * item_input).sum(dim=1)

        # Calculate additional losses
        reg_loss = self.calculate_reg_loss()
        diff_loss = self.calculate_diff_loss(u_content_transformed, v_content_transformed, u_pref_cf, v_pref_cf)

        # Return output and losses
        return output, reg_loss, diff_loss




# --- Training and evaluation ---
def train(model, data, optimizer, batch_size, num_epochs, neg, item_warm, dataset, device):
    model.train()
    criterion = nn.MSELoss()

    user_list = data['user_list']
    item_list = data['item_list']

    for epoch in range(num_epochs):
        total_loss = 0
        total_prediction_loss = 0
        total_reg_loss = 0
        total_diff_loss = 0

        # Negative Sampling
        user_array, item_array, target_array = negative_sampling(user_list, item_list, neg, item_warm)
        random_idx = np.random.permutation(user_array.shape[0])
        n_targets = len(random_idx)
        data_batches = [(n, min(n + batch_size, n_targets)) for n in range(0, n_targets, batch_size)]

        for start, end in tqdm(data_batches, desc=f"Epoch {epoch+1}"):
            batch_indices = random_idx[start:end]
            batch_users = user_array[batch_indices]
            batch_items = item_array[batch_indices]
            batch_targets = target_array[batch_indices]

            u_pref = torch.tensor(data['u_pref'][batch_users], device=device, dtype=torch.float32)
            v_pref = torch.tensor(data['v_pref'][batch_items], device=device, dtype=torch.float32)

            u_content = None
            v_content = None

            if dataset == 'CiteULike':
                v_content = torch.tensor(data['v_content'][batch_items], device=device, dtype=torch.float32)
            elif dataset == 'LastFM':
                u_content = torch.tensor(data['u_content'][batch_users], device=device, dtype=torch.float32)
            elif dataset == 'XING':
                u_content = torch.tensor(data['u_content'][batch_users], device=device, dtype=torch.float32)
                v_content = torch.tensor(data['v_content'][batch_items], device=device, dtype=torch.float32)

            targets = torch.tensor(batch_targets, device=device, dtype=torch.float32)

            # Forward pass and loss calculation
            optimizer.zero_grad()
            predictions, reg_loss, diff_loss = model(u_pref, u_content, v_pref, v_content)
            prediction_loss = criterion(predictions, targets)
            loss = prediction_loss + reg_loss + diff_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_prediction_loss += prediction_loss.item()
            total_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            total_diff_loss += diff_loss.item() if isinstance(diff_loss, torch.Tensor) else diff_loss

        # Print epoch statistics
        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}, "
              f"Prediction Loss: {total_prediction_loss:.4f}, "
              f"Regularization Loss: {total_reg_loss:.4f}, "
              f"Difference Loss: {total_diff_loss:.4f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CiteULike')
    parser.add_argument('--latent-dim', type=int, default=200)
    parser.add_argument('--content-dim', type=int, default=300)
    parser.add_argument('--output-dim', type=int, default=200)
    parser.add_argument('--num-experts', type=int, default=5)
    parser.add_argument('--random-prob', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--neg', type=int, default=5, help='Number of negative samples per positive sample')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data)
    item_warm = np.unique(data['item_list'])

    model = Heater(
        latent_dim=args.latent_dim,
        content_dim=args.content_dim,
        output_dim=args.output_dim,
        num_experts=args.num_experts,
        random_prob=args.random_prob
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, data, optimizer, args.batch_size, args.epochs, args.neg, item_warm, args.data, device)



if __name__ == "__main__":
    main()
