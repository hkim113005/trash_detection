import torch
from ultralytics import YOLO

class Model():
    def __init__(self):
        torch.cuda.empty_cache()

        self.model = YOLO("/work/home/jae/cvat/yolov10/ultralytics/cfg/models/v8/yolov8-schi-dc.yaml")

    def train(self, data):
        self.model.train(
            data=data,              # Path to the dataset configuration file
            epochs=100,              # Number of epochs to train for
            time=None,              # Maximum training time
            patience=100,            # Epochs to wait for no observable improvement for early stopping of training
            batch=16,               # Number of images per batch
            imgsz=640,              # Size of input images as integer
            device=3,               # Device to run on, i.e. cuda device=0 
            project="/work/home/jae/trash_detection/results",
            optimizer='auto',       # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
            seed=0,                 # Random seed for reproducibility
            close_mosaic=10,        # Disables mosaic data augmentation in the last N epochs
            freeze=None,            # Freezes the first N layers
            lr0=0.01,               # Initial learning rate 
            lrf=0.01,               # Final learning rate (lr0 * lrf)
            momentum=0.937,         # Momentum factor for SGD or beta1 for Adam optimizers
            weight_decay=0.0005,    # l2 regularization term, penalizing large weights to prevent overfitting
            warmup_epochs=3.0,      # Number of epochs for learning rate warmup
            warmup_momentum=0.8,    # Initial momentum for warmup phase
            warmup_bias_lr=0.1,     # Learning rate for bias parameters during warmup phase
            box=7.5,                # Weight of box loss
            cls=0.5,                # Weight of classification loss
            dfl=1.5,                # Weight of distribution focal loss
            dropout=0.0,            # Use dropout regularization
            cache=False
        )

model = Model()

model.train("/work/home/jae/trash_detection/dataset_det/data.yaml")