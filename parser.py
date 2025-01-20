import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Evaluation Parameters
    parser.add_argument('--topN', type=str, default='[5, 10, 20, 30, 50, 100]', help='top N items for evaluation.')
    parser.add_argument('--gt_path', type=str, default='./data/ML25M/BPR_cv/cold_movies_rating_all_0.tsv', help='preference items of each user')

    # Data paths
    parser.add_argument('--emb_path', type=str, default='./data/ML25M/BPR_cv/BPR_ivec_0.npy', help='load item emb path')
    parser.add_argument('--user_path', type=str, default='./data/ML25M/BPR_cv/BPR_uvec_0.npy', help='load user emb path')
    parser.add_argument('--tag_emb_path', type=str, default='./data/ML25M/mv-tag-emb.npy', help='load tag emb path')

    # Model and training parameters
    parser.add_argument('--num_t_samples', type=int, default=1, help='number of time(t) samples for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for MLP')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for MLP')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=30000, help='upper epoch limit')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')

    # MLP parameters
    parser.add_argument('--model', type=str, default='MLP', help='select model(MLP, ResidualMLP, Attention)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate of MLP layer')
    parser.add_argument('--in_dims', type=str, default='[128, 64, 128]', help='the dims for item embedding')
    parser.add_argument('--tag_emb_dim', type=int, default=400, help='the dims for tag embedding')
    parser.add_argument('--time_emb_dim', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--mlp_act_func', type=str, default='tanh', help='the activation function for MLP')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for MLP: Adam, AdamW, SGD, etc.')

    # Diffusion parameters
    parser.add_argument('--noise_scale', type=float, default=0.005, help='noise scale')
    parser.add_argument('--objective', type=str, default='pred_x0', help='objective type: pred_noise, pred_x0, pred_v')
    parser.add_argument('--timesteps', type=int, default=1000, help='diffusion steps') 
    parser.add_argument('--noise_schedule', type=str, default='linear', help='the schedule for noise generating')
    # clamp_k | None | 1 | 2 | 3 |...
    parser.add_argument('--clamp_k', type=int, default=None, help='lower and upper bound for clamping distribution, if k=2, 0.9544 if k=3 0.9973')     

    return parser.parse_args()