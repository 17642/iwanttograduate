import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from multiprocessing import Pool
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
import os
#import lmdb
import torchaudio
import torch.nn.functional as F
import time
from functools import partial
import json

from torch.serialization import SourceChangeWarning

name_class_index = {
    '강제추행(성범죄)': 0,
    '강도범죄': 1,
    '절도범죄': 2,
    '폭력범죄': 3,
    '화재': 4,
    '갇힘': 5,
    '응급의료': 6,
    '전기사고': 7,
    '가스사고': 8,
    '낙상': 9,
    '붕괴사고': 10,
    '태풍-강풍': 11,
    '지진': 12,
    '도움요청': 13,
    '실내': 14
}

label_list = {
    0: '강제추행(성범죄)',
    1: '강도범죄',
    2: '절도범죄',
    3: '폭력범죄',
    4: '화재',
    5: '갇힘',
    6: '응급의료',
    7: '전기사고',
    8: '가스사고',
    9: '낙상',
    10: '붕괴사고',
    11: '태풍-강풍',
    12: '지진',
    13: '도움요청',
    14: '실내'
}

database_dir = ".\\datasets"
save_format = ".npz"

chunk_max_size = 1000

#example_data_name = "0_0_1000.npz"

label_dict_num = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,3]
sound_label = "_label.wav"

label_num_example = 3
sound_num_example = 2
sound_file_num_example = 1400

data_directory = "D:\\위급상황 음성_음향\\"
train_directory = "Training\\"
vaildation_directory = "Validation\\"

label_tag = "[라벨]"
sound_tag = "[원천]"

max_sound_length = 4.0
sampling_rate = 44100

max_pad_len = 350

maxindex = 14

CLASS_NUM = 15

MAX_FRAMES = 128

mel_transform = MelSpectrogram(
    sample_rate=sampling_rate,
    n_mels=64
)

amplitude_to_db = AmplitudeToDB()

def get_sound_file_list(label_index, innum):
    sound_file_list = []
    target_dict = data_directory+ train_directory + sound_tag +str(label_index+1)+'.'+label_list[label_index]+'_'+str(innum)+'\\'
    for filename in os.listdir(target_dict):
        sound_file_list.append(target_dict+filename)

    return sound_file_list

def make_spce_from_sound(sound, sr):
    if sr!= sampling_rate:
        sound = torchaudio.transforms.Resample(sr, sampling_rate)(sound)
    
    if sound.size(0) > 1:
        sound = sound.mean(dim=0, keepdim=True)

    sound = (sound - sound.mean()) / (sound.std() + 1e-9)

    mel_spec = mel_transform(sound)
    mel_db = amplitude_to_db(mel_spec)

    if mel_db.shape[2] > MAX_FRAMES:
        mel_db = mel_db[:, :, :MAX_FRAMES]
    elif mel_db.shape[2] < MAX_FRAMES:
        pad_size = MAX_FRAMES - mel_db.shape[2]
        mel_db = F.pad(mel_db, (0, pad_size))

    return mel_db.squeeze(0)

def make_file_list_list():
    llist = {}
    for i in range(CLASS_NUM):
        for j in range(label_dict_num[i]):
            lst = get_sound_file_list(i,j+1)
            llist[f"{i}_{j}"] = lst
    print(f"File List Got at [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")
    return llist

label_path = "_lb.json"

def process_sound_from_file_list(lst, label, fnum):
    sample_count = 0
    chunk = 0
    label_num = 0

    temporal_spectogram = []
    label_namelist = []

    for fname in lst:
        
        label_namelist.append({"index":label_num,f"file_name":fname})

        label_num +=1

        so, sr = torchaudio.load(fname)
        temporal_spectogram.append(make_spce_from_sound(so,sr))

        sample_count +=1
        if sample_count == chunk_max_size:
            save_path = database_dir+f"\\{label}_{fnum}_{chunk}_{sample_count}.npy"
            np.save(save_path,np.array(temporal_spectogram))
            chunk +=1
            sample_count = 0
            temporal_spectogram.clear()
            print(f"Saved: {save_path} at [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")
    
    save_path = database_dir+f"\\{label}_{fnum}_{chunk}_{sample_count}.npy"
    np.save(save_path,np.array(temporal_spectogram))
    print(f"Saved: {save_path} at [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]")
    temporal_spectogram.clear()
    
    label_json_path_open = f"{database_dir}\\{label}{label_path}"
    with open(label_json_path_open,'w') as json_file:
        json.dump(label_namelist, json_file, indent=4)
    
    label_namelist.clear()

THREAD_COUNT = 4

def worker(args, lst):
    key,i,j=args
    process_sound_from_file_list(lst[key],i,j)

def main():
    os.makedirs(database_dir, exist_ok=True)
    lst = make_file_list_list()

    print(f"SAMPLE: {lst['2_0'][2]}")

    tasks = [[f"{i}_{j}",i,j] for i in range(CLASS_NUM) for j in range(label_dict_num[i])]

    with Pool(processes=THREAD_COUNT) as pool:
        wk = partial(worker,lst=lst)
        pool.map(wk,tasks)



if __name__ == "__main__":
    main()




    



