import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing as prep
from scipy.sparse import lil_matrix, csr_matrix, load_npz
from tqdm import tqdm
import argparse
import datetime
import time
import pickle
import data

# --- Timer for logging ---
class Timer:
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        usage is:
            $ timer = utils.timer('name').tic()
            $ timer.toc('process A').tic()


        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, Timer._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    @staticmethod
    def _format(s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s



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

def standardize_2(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std
    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
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


def load_data(data_name):
    timer = Timer(name='main').tic()
    data_path = './data/' + data_name
    item_content_file = data_path + '/item_features.txt'
    train_file = data_path + '/train.csv'
    test_file = data_path + '/test.csv'
    vali_file = data_path + '/vali.csv'

    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        num_user = info['num_user']
        num_item = info['num_item']

    dat = {}
    # Load preference data
    timer.tic()
    u_pref = np.load(data_path + '/U_BPR.npy')
    v_pref = np.load(data_path + '/V_BPR.npy')

    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref

    timer.toc(f'Loaded U: {u_pref.shape}, V: {v_pref.shape}').tic()

    # Pre-process preference data
    _, dat['u_pref'] = standardize(dat['u_pref'])
    _, dat['v_pref'] = standardize_2(dat['v_pref'])

    timer.toc('Standardized U and V').tic()

    # Load and process content data
    timer.tic()
    item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
    item_content = tfidf(item_content)

    from sklearn.utils.extmath import randomized_svd
    u, s, _ = randomized_svd(item_content, n_components=300, n_iter=5)
    item_content = u * s

    _, item_content = standardize(item_content)
    dat['v_content'] = item_content
    timer.toc(f'Loaded item content matrix: {item_content.shape}').tic()

    # Load train, test, and validation split data
    timer.tic()
    train_data = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train_data['uid'].values
    dat['item_list'] = train_data['iid'].values

    test_data = data.load_eval_data(test_file)
    dat['test_eval'] = test_data

    vali_data = data.load_eval_data(vali_file)
    dat['vali_eval'] = vali_data

    timer.toc(f'Loaded train data: {train_data.shape} and evaluation data').tic()

    return dat


def calculate_precision_recall(preds, ground_truth, k):
    """
    Precision@k and Recall@k calculation.
    """
    topk_preds = preds[:, :k]
    relevant_items = torch.sum(ground_truth, dim=1)
    
    true_positive = torch.sum(ground_truth.gather(1, topk_preds), dim=1)
    precision = true_positive / k
    recall = true_positive / relevant_items.clamp(min=1)  # Avoid division by zero
    
    return precision.mean().item(), recall.mean().item()

def calculate_ndcg(preds, ground_truth, k):
    """
    NDCG@k calculation.
    """
    topk_preds = preds[:, :k]
    ideal_sorted = torch.sort(ground_truth, descending=True, dim=1).values
    ideal_dcg = ((ideal_sorted[:, :k] / torch.log2(torch.arange(2, k + 2).float().to(preds.device))).sum(dim=1))

    dcg = ((ground_truth.gather(1, topk_preds) / torch.log2(torch.arange(2, k + 2).float().to(preds.device))).sum(dim=1))
    
    ndcg = (dcg / ideal_dcg.clamp(min=1e-10)).mean().item()
    return ndcg


# --- Model definition ---
class Heater(nn.Module):
    def __init__(self, latent_dim, content_dim, output_dim, num_experts=5, random_prob=0.5, dropout=0.5, alpha=0.0001, beta=0.0001):
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
        self.user_content_layer = nn.Linear(output_dim, output_dim)
        self.item_content_layer = nn.Linear(output_dim, output_dim)

        # Additional layer to mimic `dense_batch_fc_tanh`
        self.tanh_fc_layer = nn.Sequential(nn.Linear(output_dim, output_dim),
                                           nn.BatchNorm1d(output_dim),
                                           nn.Tanh()
                                           )

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
        u_content_last, v_content_last = None, None

        # MoE & Randomized Training : Choose between CF or content embeddings
        # Randomized filter indicator (0 or 1)

        if dataset == 'CiteULike':
            v_content_expert = self.transform_content(v_content, self.gate_item, self.experts_item)
            v_content_last = self.item_content_layer(v_content_expert)

            dropout_item_indicator = (torch.rand(v_content_last.size(0), 1, device=v_content_last.device) < self.random_prob).float()

        elif dataset == 'LastFM':
            u_content_expert = self.transform_content(u_content, self.gate_user, self.experts_user)
            u_content_last = self.user_content_layer(u_content_expert)

            dropout_user_indicator = (torch.rand(u_content_last.size(0), 1, device=u_content_last.device) < self.random_prob).float()

        elif dataset == 'XING':
            u_content_expert = self.transform_content(u_content, self.gate_user, self.experts_user)
            v_content_expert = self.transform_content(v_content, self.gate_item, self.experts_item)

            u_content_last = self.user_content_layer(u_content_expert)
            v_content_last = self.item_content_layer(v_content_expert)

            dropout_user_indicator = (torch.rand(u_content_last.size(0), 1, device=u_content_last.device) < self.random_prob).float()
            dropout_item_indicator = (torch.rand(v_content_last.size(0), 1, device=v_content_last.device) < self.random_prob).float()


        diff_loss = 0

        if u_content_last is None:
            u_final = u_pref
        else:
            u_final = u_pref * dropout_user_indicator + u_content_last * (1 - dropout_user_indicator)

        if v_content_last is None:
            v_final = v_pref
        else:
            v_final = v_pref * dropout_item_indicator + v_content_last * (1 - dropout_item_indicator)


        # Additional dense layer with batch normalization and tanh activation
        u_final = self.tanh_fc_layer(u_final)
        v_final = self.tanh_fc_layer(v_final)

        # Compute the output prediction
        output = torch.matmul(u_final, v_final.T)         # [num_user, num_item, 200] → [num_user, num_items]

        # Calculate additional losses
        reg_loss = self.calculate_reg_loss()
        diff_loss = self.calculate_diff_loss(u_content_last, v_content_last, u_pref, v_pref)

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


def evaluate(model, data, batch_size, device, recall_k=[10, 20, 30, 50]):
    model.eval()
    v_content = data.get('v_content')

    # Prepare evaluation data
    eval_data = data['test_eval']
    eval_batches = eval_data.eval_batch
    total_recall, total_precision, total_ndcg = [0] * len(recall_k), [0] * len(recall_k), [0] * len(recall_k)

    idcg_array = 1 / np.log2(np.arange(1, 101) + 1)
    idcg_table = np.array([np.sum(idcg_array[:i + 1]) for i in range(100)])

    preds_all_batches = []

    with torch.no_grad():
        for batch, (eval_start, eval_stop) in enumerate(eval_batches):
            # Get batch user and item data
            eval_users = eval_data.test_user_ids[eval_start:eval_stop]
            eval_items = eval_data.test_item_ids

            # Convert to tensor
            u_pref = torch.tensor(data['u_pref'][eval_users], device=device, dtype=torch.float32)
            v_pref = torch.tensor(data['v_pref'][eval_items], device=device, dtype=torch.float32)
            v_content_batch = torch.tensor(v_content[eval_items], device=device, dtype=torch.float32)

            # Get predictions
            predictions, _, _ = model(u_pref, None, v_pref, v_content_batch)
            preds_all_batches.append(predictions.cpu().numpy())

        # Concatenate all prediction batches
        preds_all = np.concatenate(preds_all_batches)

        # Filter non-zero targets
        y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]
        y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]
        preds_all = preds_all[y_nz, :]

        # Create ground truth matrix
        row_indices = np.repeat(np.arange(len(y_nz)), [len(r) for r in eval_data.R_test_inf.rows[y_nz]])
        col_indices = np.concatenate(eval_data.R_test_inf.rows[y_nz])
        ground_truth = csr_matrix((np.ones_like(col_indices), (row_indices, col_indices)),
                                    shape=(len(y_nz), predictions.size(1)))

        recall = []
        precision = []
        ndcg = []

        # Loop over each recall threshold
        for at_k in recall_k:
            preds_k = preds_all[:, :at_k]

            # Create sparse matrix for predictions
            y = eval_data.R_test_inf[y_nz, :]
            pred_sparse = lil_matrix(y.shape)
            for idx, preds in enumerate(preds_k):
                pred_sparse.rows[idx] = preds
                pred_sparse.data[idx] = [1] * len(preds)

            # Convert to CSR format
            pred_sparse = pred_sparse.tocsr()

            # Calculate overlap between predictions and ground truth
            overlap = y.multiply(pred_sparse)

            # Compute recall and precision
            recall.append(np.mean(overlap.sum(axis=1) / y.sum(axis=1).clip(min=1e-10)))
            precision.append(np.mean(overlap.sum(axis=1) / at_k))

            # Calculate NDCG
            overlap_coo = overlap.tocoo()
            rows, cols = overlap_coo.row, overlap_coo.col
            y_csr = y.tocsr()

            # Get DCG values
            dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
            dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)

            # Calculate IDCG based on ground truth size
            idcg = y.sum(axis=1).A1 - 1
            idcg[idcg >= at_k] = at_k - 1
            idcg = idcg_table[idcg.astype(int)]
            ndcg.append(np.mean(dcg / idcg.clip(min=1e-10)))

        return recall, precision, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CiteULike')
    parser.add_argument('--latent-dim', type=int, default=200)
    parser.add_argument('--content-dim', type=int, default=300)
    parser.add_argument('--output-dim', type=int, default=200)
    parser.add_argument('--num-experts', type=int, default=5)
    parser.add_argument('--random-prob', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--neg', type=int, default=5, help='Number of negative samples per positive sample')
    parser.add_argument('--recall-k', type=int, nargs='+', default=[10, 20, 30, 50], 
                        help='List of thresholds for Recall@K evaluation (e.g., --recall-k 1 5 10)')
    parser.add_argument('--alpha', type=float, default=0.0001, help='diff loss weight')
    parser.add_argument('--beta', type=float, default=0.0001, help='regularization loss weight')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data)
    test_eval = data['test_eval']
    vali_eval = data['vali_eval']
    test_eval.init_tf(data['u_pref'], data['v_pref'], None, data['v_content'], args.eval_batch_size, cold_item=True)
    vali_eval.init_tf(data['u_pref'], data['v_pref'], None, data['v_content'], args.eval_batch_size, cold_item=True)

    item_warm = np.unique(data['item_list'])

    model = Heater(
        latent_dim=args.latent_dim,
        content_dim=args.content_dim,
        output_dim=args.output_dim,
        num_experts=args.num_experts,
        random_prob=args.random_prob,
        alpha=args.alpha,
        beta=args.beta
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ## Momentum Optimizing
    # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    train(model, data, optimizer, args.batch_size, args.epochs, args.neg, item_warm, args.data, device)

    evaluate(model, data, args.batch_size, device, recall_k=args.recall_k)

if __name__ == "__main__":
    main()
