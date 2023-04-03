import argparse
import logging
from pathlib import Path
import time
import datetime
import os
import pickle

from numba.core.errors import NumbaWarning
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import get_args_parser
from prepare_data import prepare_dataset
from dataset import create_dataloader
from model import SolarModel
#from postprocess import postprocess



import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


def main(args):
    print(args)
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    # Prepare Data by cleaning and preprocessing
    if args.prepare_data:
        prepare_dataset(args)
        return
    
    # Create DataLoader
    training, val, train_dataloader, val_dataloader = create_dataloader(args)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 1e-7, patience=args.patience, verbose = True, mode = "min")
    lr_logger = LearningRateMonitor()
    if args.target_mode == "multiple":
        logger = TensorBoardLogger(args.output_dir + "model_logs/{}_{}".format(args.station, args.model))
    else:
        logger = TensorBoardLogger(args.output_dir + "model_logs/{}_{}_{}".format(args.station, args.model, args.target))
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        enable_model_summary= True,
        gradient_clip_val= args.clip_grad,
        callbacks=[early_stop_callback, lr_logger],
        logger = logger,
        log_every_n_steps=10
    )
    
    # Create Model
    model = SolarModel(args).create(training)
    model.to(device)

    # Hyperparameter Tuning
    if args.param_optimize:
        if args.model == "TFT":
            # create a new study
            study = optimize_hyperparameters(
                train_dataloader,
                val_dataloader,
                model_path="Optuna",
                n_trials=200,
                max_epochs=20,
                gradient_clip_val_range=(0.01, 1.0),
                hidden_size_range=(8, 128),
                hidden_continuous_size_range=(8, 128),
                attention_head_size_range=(1, 4),
                learning_rate_range=(0.001, 0.1),
                dropout_range=(0.1, 0.3),
                trainer_kwargs=dict(limit_train_batches=30),
                reduce_on_plateau_patience=args.patience,
                use_learning_rate_finder=False,
                log_dir = "model_logs"
            )
            # save study results
            with open(args.output_dir + "study.pkl", "wb") as fout:
                pickle.dump(study, fout)

            # print best hyperparameters
            print(study.best_trial.params)

        return
    # Predict
    if args.test:
        if args.model == "base":
            preds = model.predict(val_dataloader)

        else:
            if args.target_mode == "multiple":
                path = args.output_dir + "model_logs/{}_{}".format(args.station, args.model) + "/lightning_logs/"
            else:
                path = args.output_dir + "model_logs/{}_{}_{}".format(args.station, args.model, args.target) + "/lightning_logs/"
            newest_version = max([os.path.join(path,d) for d in os.listdir(path)], key=os.path.getmtime) + "/checkpoints/"
            checkpoint = newest_version + os.listdir(newest_version)[0]

            model = model.load_from_checkpoint(checkpoint)
            preds, x = model.predict(val_dataloader, mode = "raw", return_x = True)

            interpretation = model.interpret_output(preds, reduction = "sum")
            model.plot_interpretation(interpretation)

            with open(args.output_dir + "preds.pickle","wb") as f:
                pickle.dump(preds["prediction"],f)
            f.close()
    

        with open(args.output_dir + "val.pickle","wb") as f:
            pickle.dump(val,f)
            f.close()


        #preds = postprocess(preds)

        return
    # FineTuning
    if args.finetune:
        if args.target_mode == "multiple":
            path = args.output_dir + "model_logs/{}_{}".format(args.station, args.model) + "/lightning_logs/"
        else:
            path = args.output_dir + "model_logs/{}_{}_{}".format(args.station, args.model, args.target) + "/lightning_logs/"
        newest_version = max([os.path.join(path,d) for d in os.listdir(path)], key=os.path.getmtime) + "/checkpoints"
        checkpoint = os.listdir(newest_version)[0]

        model = model.load_from_checkpoint(checkpoint)

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