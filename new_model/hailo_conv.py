from hailo_sdk_client import ClientRunner
#from hailo_platform import HailoRT
import onnx
import argparse
import numpy as np
#from hailo_sdk_client import run_quantization_from_np
from datasets import EmergencySoundDataset
from torch.utils.data import DataLoader
import random
import os

DEFAULT_TARGET_MODEL = 'converted_model.onnx'
DEFAULT_OUTPUT_MODEL = 'converted_model.hef'
DEFAULT_HAR_OUT = 'converted_model.har'
DEFAULT_OPCODE = -1
DEFAULT_MACHINE = 'hailo8l'
DEFAULT_MODEL_NAME = 'main_graph'
DEFAULT_DATASET_PATH = "valid_dataset\\"
DEFAULT_DATA_PARSE_SIZE = 360

INPUTS = ['input']
OUTPUTS = ['output']
INPUT_SHAPES = {'input':[1,1,64,128]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python hailo_conv',description='hef converter')
    parser.add_argument('-t','--target',action='store',default=DEFAULT_TARGET_MODEL,type=str)
    parser.add_argument('-o','--output',action='store',default=DEFAULT_OUTPUT_MODEL,type=str)
    parser.add_argument('-ho','--har-output',action='store',default=DEFAULT_HAR_OUT,type=str)
    parser.add_argument('-s','--opset',action='store',default=DEFAULT_OPCODE,type=int)
    parser.add_argument('-m','--machine',action='store',default=DEFAULT_MACHINE,type=str)
    parser.add_argument('-n','--model-name',action='store',default=DEFAULT_MODEL_NAME,type=str)
    parser.add_argument('-p','--data-path',action='store',default=DEFAULT_DATASET_PATH,type=str)
    parser.add_argument('-ds','--dataset-size',action='store',default=DEFAULT_DATA_PARSE_SIZE,type=int)
    
    args = parser.parse_args()
    
    try:
        model = onnx.load(args.target)
        onnx.checker.check_model(model)
        
        print("ONNX model is valid. starting convert to hef")
        
        runner = ClientRunner(hw_arch=args.machine)
        
        hn, npz = runner.translate_onnx_model(
            args.target,
            args.model_name,
            start_node_names=INPUTS,
            end_node_names=OUTPUTS,
            net_input_shapes=INPUT_SHAPES,
        )
        
        
        dataset = EmergencySoundDataset(args.data_path)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        
        sample_ind = random.sample(range(len(dataset)),args.dataset_size)
        calib_inputs = []
        
        for i in sample_ind:
            x, _ = dataset[i]
            calib_inputs.append(x)
        
        calib_array = np.concatenate(calib_inputs, axis=0)
        #run_quantization_from_np(runner,calib_data)
        print(calib_array.shape)
        calib_array = calib_array.reshape(args.dataset_size,64,128,1)
        runner.optimize(calib_array,data_type='np_array')
        
        hefs = runner.compile()
        
        with open(args.output,'wb') as f:
            f.write(hefs)
            print(f"[info] HEF FILE SAVED TO {args.output}")
            f.close()

        runner.save_har(args.har_output)
        runner.save_hef(args.output)
        
        print(f"[info] Checking Saved model...")
        print(f"[info] HEF file Exists: {os.path.exists(args.output)}, Size: {os.path.getsize(args.output)}")
        print(f"[info] Check Input Shapt/Inference/Output...")
        with HailoRT() as h:
            h.load_hef(args.output)
            output = h.infer(np.random.rand(1,1,64,128).transpose(0,2,3,1))
            print(f"Inference OK, OUTPUT shape is {output.shape}")
        
        print("[info] Model Check Finished!")
        
    except onnx.checker.ValidationError as e:
        print(f"[ERROR] ONNX model check failed.: {e}")
    except Exception as e:
        print(f"[ERROR] There is an error: {e}")
