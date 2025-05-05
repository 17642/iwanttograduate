import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from models import CNNAudioClassifier
import configs
from datasets import *
from train import collate_fn
import sys
from colorama import init
import time

class Validation(CNNAudioClassifier):
    def __init__(self, _modelPath, _dataPath, num_classes, max_Validation_length=-1):
        super().__init__(num_classes)

        self.modelPath = _modelPath
        self.dataPath = _dataPath
        self.ValidLen = max_Validation_length

        self.load_state_dict(torch.load(self.modelPath))
        print(f"{'MODEL LOADED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")

    def clearandprint(self, str):
        print(" "*64)
        sys.stdout.write("\033[1A")
        sys.stdout.flush()
        print(str)

    def validate(self, dataloader):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        if self.ValidLen == -1 or self.ValidLen>len(dataloader):
            self.ValidLen = len(dataloader)

        print(f"VALIDATION LEN SET TO {self.ValidLen}")

        print(f"{'START VALIDATION':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")
        i = 0
        st = time.time()
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                if i == 0:
                    print("-" * 32)
                    print(f"{'Sample Output and Labels:':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]\n{'OUTPUT SHAPE:':<20}{outputs.shape}\n{'LABEL SHAPE:':<20}{labels.shape}")
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                pred = outputs.argmax(dim=1)
                if i == 0:
                    print("-" * 32)
                    print(f"Sample pred and Label\n\n")
                if i < 3:
                    self.clearandprint(f"{'[Pred]':<10} {pred[0]:<10}{'[Label]':<10}{labels[0]}")
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                i+=1
                self.clearandprint(f"{i}/{self.ValidLen} {'iter':<10}...{i/(time.time()-st):.2f} iter/sec. {time.time()-st:.2f}s elapsed.")
                if self.ValidLen<i+1:
                    break
                if i<self.ValidLen:
                    sys.stdout.write("\033[1A")
                    sys.stdout.flush()
                
        print("-" * 32)
        print(f"{'VALIDATION FINISHED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")

        avg_loss = total_loss/self.ValidLen
        accuracy = correct/total

        return {"loss": avg_loss, "accuracy": accuracy}




def main():

    MODEL_PATH = ".\\checkpoints\\best_model.pth"
    DATA_PATH = ".\\valid_dataset"
    VALIDATE_BATCH = 32
    sets = -1
    if(len(sys.argv)==2):
        sets = int(sys.argv[1])

    v = Validation(MODEL_PATH, DATA_PATH, configs.NUM_CLASSES,sets)
    validate_dataset = EmergencySoundDataset(DATA_PATH)
    validate_dataloader = DataLoader(validate_dataset, batch_size=VALIDATE_BATCH , shuffle=True, num_workers=4, collate_fn=collate_fn)
    print(f"{'DATA LOADED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")
    rst = v.validate(validate_dataloader)
    print("-" * 32)
    print(f"VALIDATION RESULT:\n\n{'LOSS:':<25}{rst['loss']:>6.4f}\n{'ACCURACY:':<24}{rst['accuracy']*100:>6.2f}%")
    print("-" * 32)


if __name__ == "__main__":
    init()
    main()
