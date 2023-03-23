from pytorch_forecasting.models import Baseline, TemporalFusionTransformer, AutoRegressiveBaseModel, DeepAR

def create_model(args):
    if args.
    if args.model == "base":
        return Baseline()
    elif args.model == "TFT":
        max_epochs = args.epochs
        accelerator = "gpu"
        devices = args.device
        enable_model_summary = True
        gradient_clip_val = 0.1
        callbacks = [lr_logger, early_stop_callback]
        return TemporalFusionTransformer()
    elif args.model == "AutoRegressive":
        return AutoRegressiveBaseModel()
    