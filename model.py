
import pytorch_lightning as pl

from pytorch_forecasting.models import Baseline, TemporalFusionTransformer, AutoRegressiveBaseModel, DeepAR, RecurrentNetwork
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss, RMSE, MAE, MAPE

class SolarModel():
    def __init__(self, args):
        self.args = args
        self.loss_identify()

    def loss_identify(self):

        if self.args.target_mode == "multiple":
            if self.args.loss == "QuantileLoss":
                self.loss = [QuantileLoss(quantiles = [0.25, 0.5, 0.75]), QuantileLoss(quantiles=[0.25, 0.5,0.75])]
                self.loss = MultiLoss(self.loss)
                self.output_size = [3,3]

            elif self.args.loss == "MAE":
                self.loss = [QuantileLoss(quantiles = [ 0.5]), QuantileLoss(quantiles=[0.5])]
                self.loss = MultiLoss(self.loss)
                self.output_size = [1,1]

            elif self.args.loss == "RMSE":
                self.loss = MultiLoss(RMSE(), RMSE())
                self.output_size = [1,1]
        else:
            if self.args.loss == "QuantileLoss":
                self.loss = QuantileLoss(quantiles = [0.25, 0.5, 0.75])
                self.output_size = 3

            elif self.args.loss == "MAE":
                self.loss = MAE()
                self.output_size = 1

            elif self.args.loss == "RMSE":
                self.loss = RMSE()
                self.output_size = 1

        return self
    
    def TFT_model(self, training):
        return TemporalFusionTransformer.from_dataset(
            training,
            learning_rate = self.args.lr,
            hidden_size = self.args.hidden_size,
            lstm_layers = 2,
            attention_head_size= self.args.attention_head,
            hidden_continuous_size= self.args.hidden_continuous_size,
            output_size = self.output_size,
            loss = self.loss,
            dropout = self.args.dropout,
            log_interval = self.args.log_interval,
            reduce_on_plateau_patience= self.args.patience,
        )
    def baseline_model(self, training):
        return Baseline(
        )
    
    def ARIMA_model(self, training):
        return AutoRegressiveBaseModel(
            training,
            learning_rate= self.lr,
            loss = self.loss,
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
        