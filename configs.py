import argparse

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

    # Finetune paramaters:
    parser.add_argument('--finetune', action = "store_true",
                        help = "Finetuning model with exist checkpoint")
    parser.add_argument('--cpkt_dir', default = "model_logs", type = str)

    # Predict parameters
    parser.add_argument('--verbose', action = "store_true",
                        help = "Display prediction from model")
    parser.add_argument('--test', action = "store_true",
                        help = "Test Process")
    # Model parameters
    parser.add_argument('--model', default="TFT", type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_ver', default = 0, type = int,
                        help ="Number of version of model")
    parser.add_argument('--config', default = None,
                        help = "Add config file that include model params")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout (Default: 0.1)')
    parser.add_argument('--clip_grad', type = float, default = 0.1, metavar="NORM",
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--hidden_size', type = int, default = 32,
                        help = "Size of hidden layer of model")
    parser.add_argument('--hidden_continuous_size', type = int, default=16,
                        help = "Size of hidden continuous layer of model")
    parser.add_argument('--attention_head', type = int, default = 4,
                        help = "Number of attention head in Transformer Architecture")
    parser.add_argument('--loss', type = str, default = "QuantileLoss",
                        help = "Loss Function (Quantile Loss, RMSE, MAE)")
    parser.add_argument('--log_interval', type = int, default = 1000,
                        help = "Log Interval")
    
    # Hyperparams Optimaztion
    parser.add_argument("--param_optimize", action = "store_true",
                        help = "Find best params for model")
    # Optimization parameters
    parser.add_argument("--opt", default = "adamw", type = str, metavar = 'OPTIMIZER',
                        help = "Optimizer function (ranger, adam)")
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument("--lr", default = 1.e-2, type = int,
                        help = "learning rate of optimizer")
    parser.add_argument("--patience", default = 3, type = int,
                        help = "Patience number for Early Stopping")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--save_top_k', default = 1, type = int,
                        help = "Save k numbers of best checkpoint")

    # Dataset parameters
    parser.add_argument("--station", default = "v1", type = str, 
                        help = "Choose station (v1,v2,y6,y7) as target")
    parser.add_argument('--target', default = "power_generation", type = str,
                        help = "Choose target feature (power_generation / power_demand)")
    parser.add_argument('--target_mode', default = "single", type = str,
                        help = "Multiple target or single target (power_generation/power_demand)")
    parser.add_argument('--data_dir', default='./data/2023_devday_data/', type=str,
                        help='dataset path')
    parser.add_argument('--data_output_dir', default='./dataset/', type =str,
                        help="Dataset output path")
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--max_encoder_day', default = 30, type = int,
                        help = "Max length of encoder")
    parser.add_argument('--max_pred_day', default = 2, type = int,
                        help = "Max prediction length of encoder")
    parser.add_argument('--horizon', default = 28, type = int,
                        help = "Time horizon (28 days of february - the validation month)")
    
    # Prepare process params
    parser.add_argument("--fill_na", default = "remove", type = str,
                        help="Fill NaN value (fill/remove)")
    parser.add_argument("--normalize_data", default = True, type = bool,
                        help="Normalize dataset")
    parser.add_argument("--smooth", default = "Exponent", type = str,
                        help="Smoothing techniques (Exponent, Convolution,...)" )
    parser.add_argument("--window_size", default = 48, type = int,
                        help = "Window size")
    parser.add_argument("--interval", default = "sigma", type = str,
                        help = "Prediction Interval after smoothing")
    parser.add_argument("--impute", default = "MICE", type = str,
                        help = "Imputation way to fill NaN value")

    return parser