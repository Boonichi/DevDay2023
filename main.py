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

from configs import get_args_parser
from prepare_data import prepare_dataset
from dataset import create_dataloader
from model import SolarModel
from postprocess import postprocess


from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


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
    
    # Predict
    if args.test:
        path = args.cpkt_dir +  "/" + args.model + "/lightning_logs/"
        newest_version = max([os.path.join(path,d) for d in os.listdir(path)], key=os.path.getmtime) + "/checkpoint"
        checkpoint = os.listdir(newest_version)[0]

        model = model.load_from_checkpoint(checkpoint)
        preds = model.predict(val_dataloader)
        preds = postprocess(preds)
        with open("result.csv","w") as f:
            f.write(str(preds))
            f.close()

        return
    
    # Create DataLoader
    training, val, train_dataloader, val_dataloader = create_dataloader(args)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 1e-7, patience=5, verbose = True, mode = "min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("model_logs/{}_{}".format(args.station, args.model))
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        enable_model_summary= True,
        gradient_clip_val= args.clip_grad,
        callbacks=[early_stop_callback, lr_logger],
        logger = logger
    )

    # Create Model
    model = SolarModel(args).create(training)
    model.to(device)

    # Hyperparameter Tuning
    if args.param_optimize:
        '''if args.model == "TFT":
            # create a new study
            study = optimize_hyperparameters(
                train_dataloader,
                val_dataloader,
                model_path="optuna_test",
                n_trials=1,
                max_epochs=1,
                gradient_clip_val_range=(0.01, 1.0),
                hidden_size_range=(30, 128),
                hidden_continuous_size_range=(30, 128),
                attention_head_size_range=(1, 4),
                learning_rate_range=(0.001, 0.1),
                dropout_range=(0.1, 0.3),
                reduce_on_plateau_patience=4,
                use_learning_rate_finder=False 
            )
            # save study results
            with open("test_study.pkl", "wb") as fout:
                pickle.dump(study, fout)

            # print best hyperparameters
            print(study.best_trial.params)'''
        # find optimal learning rate
        res = trainer.tuner.lr_find(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            min_lr=1e-5,
            max_lr=1e0,
            early_stop_threshold=100,
        )
        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=True, suggest=True)
        fig.show()
        model.hparams.learning_rate = res.suggestion()

        return
    
    # FineTuning
    if args.finetune:
        path = args.cpkt_dir +  "/" + args.model + "/lightning_logs/"
        newest_version = max([os.path.join(path,d) for d in os.listdir(path)], key=os.path.getmtime) + "/checkpoint"
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