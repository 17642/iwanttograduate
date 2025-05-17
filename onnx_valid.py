import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import onnx
import onnxruntime as ort
import numpy as np
import argparse
import sys
import time
from colorama import init
import configs
import datasets
from torch.utils.data import DataLoader
import torch

DEFAULT_ONNX_PATH = "onnx_model.onnx"
DEFAULT_DATASET_PATH = "valid_dataset\\"

DEFAULT_BATCH_SIZE = 32

def main(args):
    session = ort.InferenceSession(args.onnx_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dataset = datasets.EmergencySoundDataset(args.data_path)
    dataloader = DataLoader(dataset, args.batch, shuffle=True)

    validlen = args.iteration
    if validlen == -1 or validlen > len(dataloader):
        validlen = len(dataloader)

    total_loss = 0
    correct = 0
    total = 0
    conf_matrix = np.zeros((configs.NUM_CLASSES, configs.NUM_CLASSES), dtype=int)

    if not args.silent:
        print(f"[INFO] Validation started using ONNX model: {args.onnx_path}")
        print(f"[INFO] Dataset: {args.data_path} | Batch size: {args.batch} | Iterations: {validlen}")
        print("-" * 60)

    for i, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs_np = inputs.numpy()

        outputs = session.run([output_name], {input_name: inputs_np})
        preds = np.argmax(outputs[0], axis=1)

        outputs_tensor = torch.from_numpy(outputs[0])
        loss = torch.nn.CrossEntropyLoss()(outputs_tensor, labels)

        correct += (preds == labels.numpy()).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        for s in range(len(labels)):
            conf_matrix[labels[s].item()][preds[s]] += 1

        if not args.silent and i % 10 == 0:
            print(f"[BATCH {i+1}/{validlen}] Loss: {loss.item():.4f} | Accuracy so far: {correct / total:.4f}")

        if i + 1 >= validlen:
            break

    avg_loss = total_loss / validlen
    accuracy = correct / total

    if not args.silent:
        print("-" * 60)
        print(f"[RESULT] Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
        print(f"[INFO] Confusion matrix saved to: confusion_matrix_onnx.npy")
        print("[INFO] Validation complete.")

    return {"loss": avg_loss, "accuracy": accuracy}, conf_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python onnx_valid.py',description='설명 아무거나 넣기')

    parser.add_argument('-c','--confusion-matrix',action='store',default=None,type=str)
    parser.add_argument('-s','--silent',action='store_true',default=False)
    parser.add_argument('-l','--save_log',action='store',default=None,type=str)
    parser.add_argument('-p','--onnx-path',action='store',default=DEFAULT_ONNX_PATH,type=str)
    parser.add_argument('-d','--data-path',action='store',default=DEFAULT_DATASET_PATH,type=str)
    parser.add_argument('-b','--batch',action='store',default=DEFAULT_BATCH_SIZE,type=int)
    parser.add_argument('-i','--iteration',action='store',default = -1,type=int)

    args = parser.parse_args()


    init()
    rtn, confusion_matrix = main(args)

    confusion_matrix = np.array(confusion_matrix)
    np.save("confusion_matrix_onnx.npy",confusion_matrix)

