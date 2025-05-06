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
import argparse
import numpy as np


DEFAULT_MODEL_PATH = ".\\checkpoints\\best_model.pth"
DEFAULT_DATA_PATH = ".\\valid_dataset"
DEFAULT_VALIDATE_BATCH = 32

LINE = '-'*32
BLANK = ' '*64

def print_with_ext(str):
    print(str)
    return str

class Validation(CNNAudioClassifier):
    def __init__(self, _modelPath, _dataPath, num_classes, max_Validation_length=-1, silent=False, txt_out = False):
        super().__init__(num_classes)

        self.modelPath = _modelPath
        self.dataPath = _dataPath
        self.ValidLen = max_Validation_length
        self.silent=silent
        self.outtxt = ""
        self.txt_out = txt_out

        self.load_state_dict(torch.load(self.modelPath))
        LOADSTR = f"{'MODEL LOADED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]"
        self.ptxtout(False,LOADSTR)

    def ptxtout(self,cls,str):
        if not self.silent:
            if cls:
                self.clearandprint(str)
            else:
                print(str)
        if self.txt_out:
            self.outtxt += str+'\n'

    def clearandprint(self, str):
        print(BLANK)
        sys.stdout.write("\033[1A")
        sys.stdout.flush()
        print(str)

    def validate(self, dataloader):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        conf_matrix = np.zeros((configs.NUM_CLASSES,configs.NUM_CLASSES),dtype=int)

        if self.ValidLen == -1 or self.ValidLen>len(dataloader):
            self.ValidLen = len(dataloader)

        VALSETSTR = f"VALIDATION LEN SET TO {self.ValidLen}"
        self.ptxtout(False,VALSETSTR)


        VALSTR = f"{'START VALIDATION':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]"
        self.ptxtout(False,VALSTR)
        i = 0
        with torch.no_grad():
            st = time.time()
            for inputs, labels in dataloader:
                outputs = self(inputs)
                if i == 0:
                    self.ptxtout(False,LINE)
                    OUTTXT = f"{'Sample Output and Labels:':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]\n{'OUTPUT SHAPE:':<20}{outputs.shape}\n{'LABEL SHAPE:':<20}{labels.shape}"
                    self.ptxtout(False,OUTTXT)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                pred = outputs.argmax(dim=1)

                if i == 0:
                    self.ptxtout(False,LINE)
                    self.ptxtout(False,f"Sample pred and Label\n\n")
                if i < 3:
                    self.ptxtout(True,f"{'[Pred]':<10} {pred[0]:<10}{'[Label]':<10}{labels[0]}")

                for s in range(len(labels)):
                    conf_matrix[labels[s]][pred[s]] += 1

                correct += (pred == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                i+=1
                ltstr = f"{i}/{self.ValidLen} {'iter':<10}...{i/(time.time()-st):.2f} iter/sec. {time.time()-st:.2f}s elapsed."
                if not self.silent:
                    self.clearandprint(ltstr)
                if self.ValidLen<i+1:
                    self.outtxt+=ltstr+'\n'
                    break
                if i<self.ValidLen and not self.silent:
                    sys.stdout.write("\033[1A")
                    sys.stdout.flush()
                
        self.ptxtout(False,LINE)
        self.ptxtout(False,f"{'VALIDATION FINISHED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")

        avg_loss = total_loss/self.ValidLen
        accuracy = correct/total

        return {"loss": avg_loss, "accuracy": accuracy},self.outtxt, conf_matrix


def main(args):

    out_log = ""
    txtout = False
    confmatrixout = False
    if args.output is not None:
        txtout = True

    if args.confusion_matrix is not None:
        confmatrixout = True

    v = Validation(args.model_path, args.data_path, configs.NUM_CLASSES,args.iteration,args.silent,txtout)
    validate_dataset = EmergencySoundDataset(args.data_path)
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch , shuffle=True, num_workers=4, collate_fn=collate_fn)
    DLSTR = f"{'DATA LOADED':<40}[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]"
    out_log+=DLSTR+'\n'
    if not args.silent:
        print(DLSTR)
    rst,logs, confusion_matrix = v.validate(validate_dataloader)
    out_log+=logs
    out_log+=print_with_ext(LINE)+'\n'
    out_log+=print_with_ext(f"VALIDATION RESULT:\n\n{'LOSS:':<25}{rst['loss']:>6.4f}\n{'ACCURACY:':<24}{rst['accuracy']*100:>6.2f}%")+'\n'
    out_log+=print_with_ext(LINE)+'\n'

    if txtout:
        with open(args.output,'w',encoding='utf-8') as f:
            f.write(out_log)
            f.close()

    if confmatrixout:
        #print(confusion_matrix)
        confusion_matrix = np.array(confusion_matrix)
        np.save(args.confusion_matrix,confusion_matrix)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Valitation',description='설명 아무거나 넣기')
    parser.add_argument('-o','--output',action='store',default=None,type=str)
    parser.add_argument('-m','--confusion-matrix',action='store',default=None,type=str)
    parser.add_argument('-s','--silent',action='store_true',default=False)
    parser.add_argument('-i','--iteration',action='store',default=-1,type=int)
    parser.add_argument('-p','--model-path',action='store',default=DEFAULT_MODEL_PATH,type=str)
    parser.add_argument('-p2','--data-path',action='store',default=DEFAULT_DATA_PATH,type=str)
    parser.add_argument('-b','--batch',action='store',default=DEFAULT_VALIDATE_BATCH,type=int)
    
    args = parser.parse_args()

    init()
    main(args)

