import argparse
import logging
from pathlib import Path
import time
import datetime

from numba.core.errors import NumbaWarning
import warnings

import torch
import numpy as np

from prepare_data import prepare_dataset
from dataset import create_dataloader
from model import create_model

import pytorch_lightning as pl

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args_parser():
    parser = argparse.ArgumentParser('Solar Model for forecasting task', add_help=False)
    
    # Train parameters
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--num_workers', default = 0, type=int,
                        help="Number of worker in DataLoader")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--verbose', action = "store_true",
                        help = "Display prediction from model")

    # Finetune paramaters:
    parser.add_argument('--finetune', action = "store_true",
                        help = "Finetuning model with exist checkpoint")
    parser.add_argument('--model_prefix', default = "", type = str)
    # Predict parameters
    parser.add_argument('--test', action = "store_true",
                        help = "Test Process")
    # Model parameters
    parser.add_argument('--model', default='base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--config', default = None,
                        help = "Add config file that include model params")
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input size of Solar Forecasting Model')
    parser.add_argument('--clip_grad', type = float, default = None, metavar="NORM",
                        help='Clip gradient norm (default: None, no clipping)')
    #parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
    #                    help="Layer scale initial values")
    
    # EMA related parameters
    #parser.add_argument('--model_ema', type=str2bool, default=False)
    #parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    #parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    #parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Dataset parameters
    parser.add_argument('--prepare_data', action="store_true", 
                        help = "Prepare data")
    parser.add_argument('--data_dir', default='2023_devday_data/v1', type=str,
                        help='dataset path')
    parser.add_argument('--data_output_dir', default='dataset/clean/v1.csv', type =str,
                        help="Dataset output path")
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--max_encoder_len', default = 7 * 48, type = int)
    parser.add_argument('--max_pred_len', default = 2 * 48, type = int)
    
    # Prepare process params
    parser.add_argument("--imputation", default = "mean_most_impute", type = str,
                        help ="Identify imputation techniques")
    parser.add_argument("--impute_na", default = "remove", type = str,
                        help="Remove all row that include NaN value")

    return parser

def main(args):
    print(args)
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.prepare_data:
        prepare_dataset(args)
        return

    start_time = time.time()
    train_dataloader, val_dataloader = create_dataloader(args)

    model = create_model(args)
    prediction = model.predict(val_dataloader, )
    #actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    actuals = torch.cat([y for x, (y,weight) in iter(val_dataloader)])
    print(len(actuals))
    print(len(prediction))
    score = (actuals- prediction).abs().mean().item()
    print(score)

    if args.verbose:
        pass
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Solar Model for forecasting task', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)