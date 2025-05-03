import json
import os

DEFAULT_DATASET_PATH = ".\\datasets"
JSON_ENDING = "_lb.json"

DEFAULT_CLASS_COUNT = 15
DEFAULT_SUBLABEL_COUNT_LIST = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,5]
DEFAULT_MAX_CHUNK_SIZE = 1000

#Name Structure
# Npy: {label}_{sub_label}_{chunk}_{sample_count}.npy
# Json: {label}_lb.json

#data Structure
# npylist: [[[]]](3-axis). npylist[label][sublabel][cnt] -> filename
# jsonlist: [](1-axis). jsonlist[label] -> filename

def get_last_name(s):
    parts = s.split('\\')
    return parts[len(parts)-1]

def sort_npy_key(s):
    ls = get_last_name(s)
    parts = ls.split('_')
    chunk_num = int(parts[2])

    return chunk_num

def sort_json_key(s):
    ls = get_last_name(s)
    parts = ls.split('_')
    label_num = int(parts[0])

    return label_num

def make_npy_doublelist(npylist = None, class_count = DEFAULT_CLASS_COUNT, sublabel_count_list = DEFAULT_SUBLABEL_COUNT_LIST):
    npy_list_list = []
    if npylist is None:
        print("file list is Empty.")
        return None
    for _ in range(class_count):
        npy_list_list.append([])
        for f in range(sublabel_count_list[_]):
            npy_list_list[_].append([])
    
    for lfname in npylist:
        fname = get_last_name(lfname)
        sc_split = fname.split("_")
        label = int(sc_split[0])
        sublabel = int(sc_split[1])

        npy_list_list[label][sublabel].append(fname)
    for i in range(class_count):
        for j in range(sublabel_count_list[i]):
            npy_list_list[i][j] = sorted(npy_list_list[i][j],key=sort_npy_key)
    
    return npy_list_list

def get_every_file_list(target_dict = DEFAULT_DATASET_PATH, class_count = DEFAULT_CLASS_COUNT):
    npylist = []
    jsonlist = []

    for fname in os.listdir(target_dict):
        if fname.endswith(JSON_ENDING):
            jsonlist.append(fname)
        elif fname.endswith(".npy"):
            npylist.append(fname)

    jsonlist = sorted(jsonlist, key = sort_json_key)
    
    npylist = make_npy_doublelist(npylist, class_count)

    return npylist, jsonlist

def get_count_from_sublabel(sublabelList = None, max_chunk_size = DEFAULT_MAX_CHUNK_SIZE):
    if sublabelList is None:
        print("sublabel is None")
        return None
    
    lastCt = sublabelList[len(sublabelList)-1]
    lastCt = get_last_name(lastCt)
    parts = lastCt.split('_')
    count = (len(sublabelList)-1)*max_chunk_size + int(parts[3].split('.')[0])

    return count


def get_count_from_labels(labelList = None, max_chunk_size = DEFAULT_MAX_CHUNK_SIZE):
    if labelList is None:
        print("List is None")
        return None
    
    count = 0

    for i in range(len(labelList)):
        count += get_count_from_sublabel(labelList[i],max_chunk_size)

    return count

def get_metadata_from_filename(fname):
    if fname.endswith(JSON_ENDING):
        lfname = get_last_name(fname)
        label = lfname.split('_')[0]

        return {"type": "json", "label": label}
    
    elif fname.endswith(".npy"):
        lfname = get_last_name(fname)
        parts = lfname.split('_')

        return {'type': 'npy', 'label': parts[0], 'sublabel': parts[1], 'chunk_num': parts[2], 'samples_num': parts[3].split('.')[0]}
    
    
    print("Invalid File Name")
    return None

def get_filename_and_internal_index_from_entire_list(list, label, index,max_chunk_size = DEFAULT_MAX_CHUNK_SIZE):
    lst = list[label]

    target_sublabel_num = len(lst)
    target_sublabel_count_list = []
    for i in range(target_sublabel_num):
        target_sublabel_count_list.append(get_count_from_sublabel(lst[i]))
    
    cnt = 0
    sublabel_c = 0

    for i in range(target_sublabel_num):
        if index < cnt+target_sublabel_count_list[i]:
            sublabel_c = i
            break
        cnt+=target_sublabel_count_list[i]

    internal_idx = index - cnt
    internal_chunk = int(internal_idx/max_chunk_size)

    if internal_chunk<len(lst[sublabel_c])-2:
        internal_idx = target_sublabel_count_list[sublabel_c]%max_chunk_size
    else:
        internal_idx = max_chunk_size

    filename = f"{label}_{sublabel_c}_{internal_chunk}_{internal_idx}.npy"

    return filename, (index - cnt)%max_chunk_size

def get_filename_from_label_and_index(label, index, jsonlist,data_dir=DEFAULT_DATASET_PATH):
    target = f"{data_dir}\\{jsonlist[label]}"

    ret = {}
    with open(target) as f:
        json_str = json.load(f)
        ret = json_str[index]

    return ret

def test():
    llist, json_list = get_every_file_list()
    print(get_metadata_from_filename(llist[0][0][2]))
    print(get_metadata_from_filename(json_list[0]))

    print(get_filename_and_internal_index_from_entire_list(llist,2,1002))

    print(get_count_from_labels(llist[3]))

    print(get_filename_from_label_and_index(0,10000,json_list))

if __name__ == "__main__":
    test()
    

