# %%
import torch
from models import PatchMixer
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import sys
sys.argv=['']
del sys

from data_provider.data_factory import data_provider

# %%
columns = [
    # Bid price
    "L10-BidPrice", "L9-BidPrice", "L8-BidPrice", "L7-BidPrice", "L6-BidPrice", "L5-BidPrice", "L4-BidPrice", "L3-BidPrice", "L2-BidPrice", "L1-BidPrice",
    # Ask price
    "L1-AskPrice", "L2-AskPrice", "L3-AskPrice", "L4-AskPrice", "L5-AskPrice", "L6-AskPrice", "L7-AskPrice", "L8-AskPrice", "L9-AskPrice", "L10-AskPrice",
    # Bid size
    "L10-BidSize", "L9-BidSize", "L8-BidSize", "L7-BidSize", "L6-BidSize", "L5-BidSize", "L4-BidSize", "L3-BidSize", "L2-BidSize", "L1-BidSize",
    # Ask size
    "L1-AskSize", "L2-AskSize", "L3-AskSize", "L4-AskSize", "L5-AskSize", "L6-AskSize", "L7-AskSize", "L8-AskSize", "L9-AskSize", "L10-AskSize",
    # Bid No
    "L10-BuyNo", "L9-BuyNo", "L8-BuyNo", "L7-BuyNo", "L6-BuyNo", "L5-BuyNo", "L4-BuyNo", "L3-BuyNo", "L2-BuyNo", "L1-BuyNo",
    # Ask No
    "L1-SellNo", "L2-SellNo", "L3-SellNo", "L4-SellNo", "L5-SellNo", "L6-SellNo", "L7-SellNo", "L8-SellNo", "L9-SellNo", "L10-SellNo",
    # Trade data
    "VWAP", "Volume"
]

# %%
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
parser.add_argument('--is_training', type=int, default=0, help='status')
parser.add_argument('--max_num_batches_per_epoch', type=int, default=10000, help='max_num_batches_per_epoch')
parser.add_argument('--max_training_time', type=int, default=3*24*60*60, help='max_training_time')  # 3 days
parser.add_argument('--model_id', type=str, default='PatchMixer_for_lob_for_grifols', help='model id')
parser.add_argument('--model', type=str, default='PatchMixer',
                    help='model name, options: [Autoformer, Informer, Transformer, PatchMixer]')
parser.add_argument('--recover', type=bool, default=False, help='Recover training')

# data loader
parser.add_argument('--data', type=str, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/lob', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='GRLS_lob.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='500ms',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=62, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=62, help='decoder input size')
parser.add_argument('--c_out', type=int, default=62, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--mixer_kernel_size', type=int, default=8, help='patchmixer-kernel')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_false', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
args = parser.parse_args()

# !! UPDATE WITH NEW MODELS !!
setting = "PatchMixer_for_lob_for_grifols_PatchMixer_custom_ftM_sl64_ll0_pl4_dm512_nh2_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0"

folder_path = './test_results/' + setting + '/'

num_features = len(columns)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

# %%
def _predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark):
    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    # encoder - decoder

    def _run_model():
        outputs = model(batch_x)
        if args.output_attention:
            outputs = outputs[0]
        return outputs

    outputs = _run_model()

    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

    return outputs, batch_y

# %%
test_set, test_loader = data_provider(args, "test")
model = PatchMixer.Model(args).to(device)
model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth')), strict=False)
model.eval()

# %%
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        outputs, batch_y = _predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark)

        outputs = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()


        pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
        true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

        if i % 50 == 0:
            input_batch = batch_x.detach().cpu().numpy()[0]  # First X,y of current batch
            true_batch = true[0]
            pred_batch = pred[0]
            input_batch = test_set.inverse_transform(input_batch)
            true_batch = test_set.inverse_transform(true_batch)
            pred_batch = test_set.inverse_transform(pred_batch)
            
            fig, axes = plt.subplots(nrows=num_features // 2, ncols=2, figsize=(20, 60))
            axes = axes.flatten()
            for j in range(num_features):
                gt = np.concatenate((input_batch[:, j], true_batch[:, j]), axis=0)
                pd = np.concatenate((input_batch[:, j], pred_batch[:, j]), axis=0)
                axes[j].plot(gt, label='True Value')
                axes[j].plot(pd, label='Prediction')
                axes[j].set_title(columns[j])  # Display column name as titleç
                axes[j].xaxis.set_major_locator(plt.MaxNLocator(6))  # Max 6 ticks
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, "batch_" + str(i) + '.png'), bbox_inches='tight')
            plt.close(fig)
