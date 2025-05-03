import librosa
import sklearn
#impport glob
import numpy as np
import os
import json
import re
import sys
from colorama import init

data_directory = "D:\\위급상황 음성_음향\\"
train_directory = "Training\\"
vaildation_directory = "Validation\\"

converted_data_dictionary = "D:\\converted_sound\\"

label_tag = "[라벨]"
sound_tag = "[원천]"

label_list = {
    1: "강제추행(성범죄)",
    2: "강도범죄",
    3: "절도범죄",
    4: "폭력범죄",
    5: "화재",
    6: "갇힘",
    7: "응급의료",
    8: "전기사고",
    9: "가스사고",
    10: "낙상",
    11: "붕괴사고",
    12: "태풍-강풍",
    13: "지진",
    14: "도움요청",
    15: "실내",
    16: "실외"
}

label_dict_num = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,3]
sound_label = "_label.wav"

label_num_example = 3
sound_num_example = 2
sound_file_num_example = 1400

#target_dict_example = data_directory + train_directory + sound_tag+str(label_num_example)+'.'+label_list[label_num_example]+'_'+str(sound_num_example)+'\\'
#label_dict_example = data_directory+ train_directory + label_tag+str(label_num_example)+'.'+label_list[label_num_example]+'\\'
#sound_file_example = str(label_num_example)+'.'+label_list[label_num_example]+'_'+str(sound_file_num_example)+'_'+sound_label

max_sound_length = 4.0
sampling_rate = 44100

max_pad_len = 350

maxindex = 14

#print(target_dict_example)
#print(sound_file_example)
#print(label_dict_example)
#print(target_dict_example+sound_file_example)


def make_mfcc_from_sound(sound, sr):
    mfccs = librosa.feature.mfcc(y=sound, sr=sampling_rate)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width<0:
        mfccs = mfccs[:,:max_pad_len]
    elif pad_width != 0:
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)),mode='constant')

    print(mfccs.shape)

    mfccs_t=mfccs.T
    mfccs_scaled = sklearn.preprocessing.minmax_scale(mfccs_t,feature_range=(0,1))

    return mfccs_scaled.T


def get_label_file_list(label_index):
    label_file_list = []
    target_dict = data_directory+ train_directory + label_tag+str(label_index)+'.'+label_list[label_index]+'\\'

    for filename in os.listdir(target_dict):
        label_file_list.append(target_dict+filename)

    return label_file_list


def get_sound_file_list(label_index):
    sound_file_list = []
    for _ in range(label_dict_num[label_index-1]):
        target_dict = data_directory+ train_directory + sound_tag +str(label_index)+'.'+label_list[label_index]+'_'+str(_+1)+'\\'
        for filename in os.listdir(target_dict):
            sound_file_list.append(target_dict+filename)

    return sound_file_list


def make_search_list(label_index):
    sound_file_list = get_sound_file_list(label_index)
    label_file_list = get_label_file_list(label_index)

    mfcc_list = []
    i = 1
    for label_file in label_file_list:
        print(f"{i}/{len(label_file_list)}")
        f = json.load(open(label_file,'r',encoding='utf-8'))

        filename = f['audio']['fileName']
        target = ""
        if(label_index<15):
            target = data_directory+ train_directory + sound_tag +str(label_index)+'.'+label_list[label_index]+'_1\\'+filename+sound_label
            print(target)
        else:
            matchnum = rf"{label_index}\.{label_list[label_index]}_(\d+)"
            filename_num = re.search(matchnum, filename)
            for sf in sound_file_list:
                matchnum2 = rf"{label_index}\.{label_list[label_index]}_(\d+)_label\.wav"
                target_num = re.search(matchnum2, sf)
                if filename_num==target_num:
                    target = sf

        sound, sr = librosa.load(target)
        print(sr)
        mfccs = make_mfcc_from_sound(sound, sr)
        mfcc_list.append(mfccs)
        sys.stdout.write("\033[4A")
        sys.stdout.flush()
        i+=1

    return mfcc_list


def rtn_numpy_file(mfcc_list, label_index):
    numpy_list = np.array(mfcc_list)
    filestr = f"{label_index}_{label_list[label_index]}.npy"
    np.save(filestr, numpy_list)


def main():
    for i in range(maxindex):
        print(f"Making File {i+1}/{maxindex} ...: {i+1}_{label_list[i+1]}.npy")
        rtn_numpy_file(make_search_list(i+1),i+1)
        print("Finished!")


def test():
    sd, sr = librosa.load(get_sound_file_list(5)[1])
    print(sr)
    f = json.load(open(get_label_file_list(5)[1],'r', encoding='utf-8'))
    print(f)
#test()

if __name__ == '__main__':
    init()
    main()