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
from model import create_model, opt_identify

from pytorch_forecasting.models import temporal_fusion_transformer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile




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
    parser.add_argument('--num_workers', default = 8, type=int,
                        help="Number of worker in DataLoader")
    parser.add_argument('--epochs', default=20, type=int)
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
    parser.add_argument('--model', default='TFT', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--config', default = None,
                        help = "Add config file that include model params")
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input size of Solar Forecasting Model')
    parser.add_argument('--clip_grad', type = float, default = None, metavar="NORM",
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--hidden_size', type = int, default = 16,
                        help = "Size of hidden layer of model")
    parser.add_argument('--attention_head', type = int, default = 4,
                        help = "Number of attention head in Transformer Architecture")
    
    # Optimization parameters
    parser.add_argument("--opt", default = "adam", type = str, metavar = 'OPTIMIZER',
                        help = "Optimizer function (adam, lion)")
    parser.add_argument("--lr", default = 1.e-3, type = int,
                        help = "learning rate of optimizer")
    

    # Dataset parameters
    parser.add_argument('--prepare_data', action="store_true", 
                        help = "Prepare data")
    parser.add_argument('--data_dir', default='/data/2023_devday_data/v1', type=str,
                        help='dataset path')
    parser.add_argument('--data_output_dir', default='dataset/v1.csv', type =str,
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
    parser.add_argument('--max_pred_day', default = 2, type = int)
    
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
    if args.test:
        return
    
    training, val, train_dataloader, val_dataloader = create_dataloader(args)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 1e-7, patience=5, verbose = True, mode = "min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        enable_model_summary= True,
        gradient_clip_val= args.clip_grad,
        callbacks=[early_stop_callback, lr_logger],
        logger = logger
    )
    # Optimizer
    #optimizer = opt_identify(args.opt)
    model = create_model(args, training)
    # FineTuning
    if args.finetune:
        best_model_path = trainer.checkpoint_callback.best_model_path
        print("Best Model Path",best_model_path)
        best_tft = model.load_from_checkpoint("/Users/phanvanhung/devday2023/lightning_logs/lightning_logs/version_10/checkpoints/epoch=3-step=2108.ckpt")
        preds = best_tft.predict(train_dataloader)
        with open("result.txt", "w") as f:
            f.write(str(preds))
            f.close()
        return
    # Train Process
    start_time = time.time()
    trainer.fit(
        model = model,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Solar Model for forecasting task', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)