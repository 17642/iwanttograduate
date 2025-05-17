import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import sys
from colorama import init
import time
import argparse
import numpy as np

DEFAULT_MODEL_PATH = "best_model.pth"
DEFAULT_TARGET_PATH = "converted_model.onnx"
DEFAULT_BATCH_SIZE = 1
DEFAULT_CLASS_NUM = 15
DEFAULT_OPSET_VERSION = 15

class CNNAudioClassifier(nn.Module):
    def __init__(self, num_classes,batch_size):
        super(CNNAudioClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 64, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,ceil_mode = True),                           # [B, 16, 32, 64]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # [B, 32, 32, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,ceil_mode = True),                           # [B, 32, 16, 32]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [B, 64, 16, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,ceil_mode = True),                           # [B, 64, 8, 16]
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        #x = x.unsqueeze(1)
        #print(f"unsqueeze_shape: {x.shape}")
        for i,layer in enumerate(self.conv_layers):
            x = layer(x)
            print(f"after layer {i} ({layer}): {x.shape}")
        #x = self.conv_layers(x)
        x = x.view(x.size(0), 64 * 8 * 16)  # [B, 8192]
        x = self.dropout(x)
        out =  self.classifier(x)
        print(f"output_shape: {out.shape}")
        return out


def main(args):
    model = CNNAudioClassifier(num_classes = args.class_num,batch_size = args.batch_size)
    model.eval()
    
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    dummy_input = torch.randn(args.batch_size,1,64,128)
    
    torch.onnx.export(
        model,
        dummy_input,
        args.target_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=args.opset_version,
        do_constant_folding=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python3 new_model_make.py", description="model_make")
    parser.add_argument('-m','--model-path', default=DEFAULT_MODEL_PATH,type=str)
    parser.add_argument('-c','--class-num', default=DEFAULT_CLASS_NUM,type=int)
    parser.add_argument('-t','--target-path', default=DEFAULT_TARGET_PATH,type=str)
    parser.add_argument('-b','--batch-size', default=DEFAULT_BATCH_SIZE,type=str,help="unused")
    parser.add_argument('-o','--opset-version',default=DEFAULT_OPSET_VERSION,type=int)
    
    args = parser.parse_args()
    main(args)
    
