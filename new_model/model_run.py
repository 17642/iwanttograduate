import numpy as np
import threading
import time
import sounddevice as sd
from multiprocessing import Process
#import hailo_runtime
#from hailo_platform import HailoStreamInterface, HailoStreamParams, HailoDevice, HailoInputVStream, HailoOutputVStream
#from hailo_model_zoo.core.infer import run_inference
#from hailo_model_zoo.utils.parse_utils import parse_hef
# hailo_platform
#from hailo_platform.pyhailort.pyhailort import HEF,HailoStreamInterface, InputVStream, OutputVStream,VDevice,InferModel
#import hailo
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)


MODEL_PATH = "converted_model.hef"
INPUT_SHAPE = [1,64,128,1]
device = VDevice()



hef = HEF(MODEL_PATH)
configured_network_group = device.configure(hef)

network_group = configured_network_group[0]
network_group.activate()

#input_vstreams = InputVStream(network_group)
#output_vstreams = OutputVStream(network_group)

input_info = network_group.get_input_vstream_infos()[0]
output_info = network_group.get_output_vstream_infos()[0]

input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

#input_stream = streams.inputs[0]
#output_stream = streams.output[0]
	
input_shape = input_info.shape
#input_dtype = input_info.dtype
input_dtype = 'float32'	
random_input = np.random.rand(*input_shape).astype(input_dtype).flatten()
num_of_samples = 1
print(f"INPUT SHAPE: {input_shape}  INPUT TYPE: {input_dtype}")
print(f"INPUT INFO: {input_info}")
print(f"OUTPUT INFO: {output_info.name}")

output = None
network_group_params = network_group.create_params()
input_stream = InputVStreams(input_vstreams_params = input_vstreams_params, configured_network = network_group)
output_stream = OutputVStreams(output_vstreams_params=output_vstreams_params, configured_network = network_group)

dataset = (np.random.randint(0,255,(1,64,128,1)).astype(np.float32))/255.0
dataset2 = np.zeros((1,64,128,1)).astype(np.float32) # CREATE RANDOM DATASETS
print(dataset)

with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    input_data = {input_info.name: dataset}
    with network_group.activate(network_group_params):
        infer_results = infer_pipeline.infer(input_data)
        #print('Stream output shape is {}'.format(infer_results[output_info.name].shape))

def send(configured_network, num_frames):
    configured_network.wait_for_activation(1000)

    #dataset = (np.random.randint(0,255,(1,64,128,1)).astype(np.float32))/255.0
    vstreams_params = InputVStreamParams.make(configured_network)
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for _ in range(num_frames):
            for vstream, buff in vstream_to_buffer.items():
                vstream.send(buff)

def recv(configured_network, vstreams_params, num_frames):
    configured_network.wait_for_activation(1000)
    data = None
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_frames):
            for vstream in vstreams:
                data = vstream.recv()
                rt = 0 
                mx = 0
                print(data)
                for i in range(15):
                    mx = max(mx,data[i])
                    if mx == data[i]:
                        rt = i
                print(f"index {rt+1}: {mx}")
						
                

def recv_all(configured_network, num_frames):
    vstreams_params_groups = OutputVStreamParams.make_groups(configured_network)
    recv_procs = []
    for vstreams_params in vstreams_params_groups:
        proc = Process(target=recv, args=(configured_network, vstreams_params, num_frames))
        proc.start()
        recv_procs.append(proc)
    for proc in recv_procs:
        proc.join()
        

                
num_of_frames=1


send_process = Process(target=send, args=(network_group, num_of_frames))
recv_process = Process(target=recv_all, args=(network_group, num_of_frames))
recv_process.start()
send_process.start()
print('Starting streaming (num_of_frames={})'.format( num_of_frames))
with network_group.activate(network_group_params):
    send_process.join()
    recv_process.join()
print('Done')



device.release()


