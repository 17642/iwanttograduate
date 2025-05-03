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
    def __init__(self, _modelPath, _dataPath, num_classes):
        super().__init__(num_classes)

        self.modelPath = _modelPath
        self.dataPath = _dataPath

        self.load_state_dict(torch.load(self.modelPath))
        print(f"{'MODEL LOADED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")

    def validate(self, dataloader):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        print(f"{'START VALIDATION':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")
        i = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                if i == 0:
                    print("-" * 32)
                    print(f"Sample Output and Labels:\n{'OUTPUT SHAPE:':<20}{outputs.shape}\n{'LABEL SHAPE:':<20}{labels.shape}")
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                pred = outputs.argmax(dim=1)
                if i == 0:
                    print("-" * 32)
                    print(f"Sample pred and Label\n\n")
                if i < 3:
                    print(" "*32)
                    sys.stdout.write("\033[1A")
                    sys.stdout.flush()
                    print(f"{'[Pred]':<10} {pred[0]:<10}{'[Label]':<10}{labels[0]}")
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                i+=1
                print(f"{i}/{len(dataloader)} iter")
                if i<len(dataloader):
                    sys.stdout.write("\033[1A")
                    sys.stdout.flush()
        print("-" * 32)
        print(f"{'VALIDATION FINISHED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")

        avg_loss = total_loss/len(dataloader)
        accuracy = correct/total

        return {"loss": avg_loss, "accuracy": accuracy}




def main():

    MODEL_PATH = ".\\checkpoints\\best_model.pth"
    DATA_PATH = ".\\valid_dataset"
    VALIDATE_BATCH = 32

    v = Validation(MODEL_PATH, DATA_PATH, configs.NUM_CLASSES)
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