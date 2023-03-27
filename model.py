
import pytorch_lightning as pl

from pytorch_forecasting.models import Baseline, TemporalFusionTransformer, AutoRegressiveBaseModel, DeepAR, RecurrentNetwork
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss, RMSE, MAE, MAPE, MultivariateNormalDistributionLoss

class SolarModel():
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.hidden_size = args.hidden_size
        self.hidden_continuous_size = args.hidden_continuous_size
        self.attention_head = args.attention_head
        if args.loss == "QuantileLoss":
            self.loss = [QuantileLoss(quantiles = [0.25, 0.5, 0.75]), QuantileLoss(quantiles=[0.25, 0.5,0.75])]
            self.loss = MultiLoss(self.loss)
        elif args.loss == "MAE":
            self.loss = MultiLoss(MAE)
        self.log_interval = args.log_interval
        self.patience = args.patience
        self.opt = args.opt
        self.dropout = args.dropout

    def TFT_model(self, training):
        return TemporalFusionTransformer.from_dataset(
            training,
            learning_rate = self.lr,
            hidden_size = self.hidden_size,
            attention_head_size= self.attention_head,
            hidden_continuous_size= self.hidden_continuous_size,
            output_size = [3,3],
            loss = self.loss,
            dropout = self.dropout,
            log_interval = self.log_interval,
            reduce_on_plateau_patience=self.patience,
        )
    def baseline_model(self, training):
        return Baseline()
    
    def ARIMA_model(self, training):
        return AutoRegressiveBaseModel(
            training,
            learning_rate= self.lr,
            loss = MAE(),
            reduce_on_plateau_patience=self.patience,
            log_interval = self.log_interval,
            optimizer = self.opt
        )
    
    def DeepAR_model(self, training):
        
        return DeepAR.from_dataset(
            training,
            learning_rate = self.lr,
            hidden_size= self.hidden_size,
            dropout=self.dropout,
            loss= MultiLoss(
                [MultivariateNormalDistributionLoss(rank=30), MultivariateNormalDistributionLoss(rank = 30)]
            )
        )
    
    def RNN_model(self, training):
        
        return RecurrentNetwork()
    
    def create(self, training):
        models = {
            "TFT" : self.TFT_model,
            "base" : self.baseline_model,
            "ARIMA" : self.ARIMA_model,
            "DeepAR" : self.DeepAR_model,
            "RNN" : self.RNN_model 
        }
        return models[self.args.model](training)
        