import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing as prep
from scipy.sparse import lil_matrix, coo_matrix
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
    x_idf = np.log(x.shape[0] - 1) - np.log(1 + np.sum(x > 0, axis=0))
    x_tf = lil_matrix(x).tolil()
    x_tf.data = np.log(x_tf.data + 1)
    x_tfidf = x_tf @ lil_matrix(np.diag(x_idf))
    return x_tfidf

def standardize(x):
    scaler = prep.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_scaled = np.clip(x_scaled, -5, 5)
    return scaler, x_scaled



# --- Model definition ---
class Heater(nn.Module):
    def __init__(self, latent_dim, content_dim, output_dim, num_experts=5, random_prob=0.5):
        super(Heater, self).__init__()
        self.num_experts = num_experts
        self.random_prob = random_prob

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

    def forward(self, u_pref, u_content, v_pref, v_content):
        # Apply Mixture of Experts to transform content embeddings
        u_content_transformed = self.transform_content(u_content, self.gate_user, self.experts_user)
        v_content_transformed = self.transform_content(v_content, self.gate_item, self.experts_item)

        # Randomized Training: Choose between CF or content embeddings
        if np.random.rand() < self.random_prob:
            user_input = self.user_cf_layer(u_pref)
            item_input = self.item_cf_layer(v_pref)
        else:
            user_input = u_content_transformed
            item_input = v_content_transformed

        # Compute the output prediction
        output = (user_input * item_input).sum(dim=1)
        return output



# --- Training and evaluation ---
def train(model, data, optimizer, criterion, batch_size, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        indices = np.random.permutation(len(data['user_list']))
        batches = [(i, i + batch_size) for i in range(0, len(indices), batch_size)]

        for start, end in tqdm(batches, desc=f"Epoch {epoch+1}"):
            batch_indices = indices[start:end]
            u_pref = torch.tensor(data['u_pref'][batch_indices], device=device, dtype=torch.float32)
            v_pref = torch.tensor(data['v_pref'][batch_indices], device=device, dtype=torch.float32)
            u_content = torch.tensor(data['u_content'][batch_indices], device=device, dtype=torch.float32)
            v_content = torch.tensor(data['v_content'][batch_indices], device=device, dtype=torch.float32)
            targets = torch.tensor(data['target'][batch_indices], device=device, dtype=torch.float32)

            optimizer.zero_grad()
            predictions = model(u_pref, u_content, v_pref, v_content)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CiteULike')
    parser.add_argument('--latent-dim', type=int, default=200, help='Latent dimension of CF embeddings')
    parser.add_argument('--content-dim', type=int, default=300, help='Dimension of content embeddings')
    parser.add_argument('--output-dim', type=int, default=200, help='Output embedding dimension')
    parser.add_argument('--num-experts', type=int, default=5, help='Number of experts in the mixture model')
    parser.add_argument('--random-prob', type=float, default=0.5, help='Probability to use CF embeddings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    data = load_data(args.data)
    model = Heater(
        latent_dim=args.latent_dim,
        content_dim=args.content_dim,
        output_dim=args.output_dim,
        num_experts=args.num_experts,
        random_prob=args.random_prob
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train(model, data, optimizer, criterion, args.batch_size, args.epochs, device)

def load_data(data_name):
    timer = Timer(name='Data Loading')
    timer.tic()

    # Load user-item interaction data
    data_path = f'./data/{data_name}'
    u_pref = np.load(f'{data_path}/U_BPR.npy')
    v_pref = np.load(f'{data_path}/V_BPR.npy')

    # Load content features
    item_content_file = f'{data_path}/item_features.txt'
    item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
    item_content = tfidf(item_content)

    # Apply dimensionality reduction (e.g., randomized SVD)
    from sklearn.utils.extmath import randomized_svd
    u, s, _ = randomized_svd(item_content, n_components=300, n_iter=5)
    item_content = u * s

    # Standardize data
    _, item_content = standardize(item_content)
    _, u_pref = standardize(u_pref)
    _, v_pref = standardize(v_pref)

    # Load train and target data
    train = pd.read_csv(f'{data_path}/train.csv', dtype=np.int32)
    target = np.ones(len(train))  # Replace with actual target data if available

    data = {
        'u_pref': u_pref,
        'v_pref': v_pref,
        'u_content': u_pref,  # Placeholder: Update if user content is available
        'v_content': item_content,
        'user_list': train['uid'].values,
        'item_list': train['iid'].values,
        'target': target
    }

    timer.toc('Data loaded successfully.')
    return data

if __name__ == "__main__":
    main()
