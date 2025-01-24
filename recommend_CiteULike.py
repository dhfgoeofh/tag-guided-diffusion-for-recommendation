import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l2_norm(para):
    return torch.sum(torch.square(para))


def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    """
    Fully connected layer with optional batch normalization and Tanh activation.
    """
    linear = nn.Linear(x.shape[1], units)
    if do_norm:
        batch_norm = nn.BatchNorm1d(units)
        x = batch_norm(x)
    x = torch.tanh(linear(x))
    return x, l2_norm(linear.weight) + l2_norm(linear.bias)


def dense_fc(x, units):
    """
    Fully connected layer without activation.
    """
    linear = nn.Linear(x.shape[1], units)
    x = linear(x)
    return x, l2_norm(linear.weight) + l2_norm(linear.bias)


class Heater(nn.Module):
    def __init__(self, latent_rank_in, user_content_rank, item_content_rank,
                 model_select, rank_out, reg, alpha, dim):
        super(Heater, self).__init__()

        self.rank_in = latent_rank_in  # input embedding dimension
        self.phi_u_dim = user_content_rank  # user content dimension
        self.phi_v_dim = item_content_rank  # item content dimension
        self.model_select = model_select  # model architecture
        self.rank_out = rank_out  # output dimension
        self.reg = reg
        self.alpha = alpha
        self.dim = dim

        # Define layers and parameters
        self.user_layers = nn.ModuleList()
        self.item_layers = nn.ModuleList()

        for hid in model_select:
            self.user_layers.append(nn.Linear(self.phi_u_dim, hid))
            self.item_layers.append(nn.Linear(self.phi_v_dim, hid))

        self.user_output = nn.Linear(model_select[-1], rank_out)
        self.item_output = nn.Linear(model_select[-1], rank_out)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, Uin, Vin, Ucontent=None, Vcontent=None, is_training=True):
        reg_loss = 0

        if self.phi_u_dim > 0 and Ucontent is not None:
            for layer in self.user_layers:
                Ucontent = F.tanh(layer(Ucontent))
                reg_loss += l2_norm(layer.weight) + l2_norm(layer.bias)
            U_embedding = self.user_output(Ucontent)
        else:
            U_embedding = Uin

        if self.phi_v_dim > 0 and Vcontent is not None:
            for layer in self.item_layers:
                Vcontent = F.tanh(layer(Vcontent))
                reg_loss += l2_norm(layer.weight) + l2_norm(layer.bias)
            V_embedding = self.item_output(Vcontent)
        else:
            V_embedding = Vin

        # Regularization loss
        reg_loss *= self.reg

        # Compute predictions
        preds = torch.sum(U_embedding * V_embedding, dim=1)
        return preds, reg_loss


def main():
    # Example configuration
    latent_rank_in = 100
    user_content_rank = 50
    item_content_rank = 50
    model_select = [64, 32]
    rank_out = 10
    reg = 0.01
    alpha = 0.1
    dim = 5

    # Example input data
    Uin = torch.randn((128, latent_rank_in))
    Vin = torch.randn((128, latent_rank_in))
    Ucontent = torch.randn((128, user_content_rank))
    Vcontent = torch.randn((128, item_content_rank))

    # Initialize and forward pass through the model
    heater = Heater(latent_rank_in, user_content_rank, item_content_rank,
                    model_select, rank_out, reg, alpha, dim)

    preds, reg_loss = heater(Uin, Vin, Ucontent, Vcontent)
    print("Predictions:", preds)
    print("Regularization Loss:", reg_loss)


if __name__ == "__main__":
    main()
