import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from configs import SAMPLE_RATE, NUM_MELS, MAX_FRAMES


def detect_index(name):
    index = name.split('.')[1].split('_')[0]
    if index == '강제추행(성범죄)':
        index = 0
    elif index == '강도범죄':
        index = 1
    elif index == '절도범죄':
        index = 2
    elif index == '폭력범죄':
        index = 3
    elif index == '화재':
        index = 4
    elif index == '갇힘':
        index = 5
    elif index == '응급의료':
        index = 6
    elif index == '전기사고':
        index = 7
    elif index == '가스사고':
        index = 8
    elif index == '낙상':
        index = 9
    elif index == '붕괴사고':
        index = 10
    elif index == '태풍-강풍':
        index = 11
    elif index == '지진':
        index = 12
    elif index == '도움요청':
        index = 13
    elif index == '실내':
        index = 14
    else:
        raise ValueError(f"Unknown category in filename: {name}")
    return index


class EmergencySoundDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.mel_spectrogram = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=NUM_MELS)
        self.amplitude_to_db = AmplitudeToDB()

        for file in os.listdir(data_dir):
            #print(f"DATAVIEW {file}")
            if file.endswith('.wav'):
                base_name = os.path.splitext(file)[0]+'.'+os.path.splitext(file)[1]
                json_path = os.path.join(data_dir, base_name[:-11] + '.json')
                if os.path.exists(json_path):
                    self.data.append((os.path.join(data_dir, file), json_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, json_path = self.data[idx]

        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        file_name = meta["audio"]["fileName"]
        label = detect_index(file_name)

        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        # ** 중요: 여기서 모든 오디오를 mono로 변환 **
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 정규화
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        # Mel-spectrogram 계산
        mel_spec = self.mel_spectrogram(waveform)
        mel_db = self.amplitude_to_db(mel_spec)

        # 시간 축 패딩 또는 자르기
        if mel_db.shape[2] > MAX_FRAMES:
            mel_db = mel_db[:, :, :MAX_FRAMES]
        elif mel_db.shape[2] < MAX_FRAMES:
            pad_size = MAX_FRAMES - mel_db.shape[2]
            mel_db = F.pad(mel_db, (0, pad_size))

        #print(f"DATA LOADED WITH: {json_path}")
        
        mel_db = mel_db.unsqueeze(1)

        return mel_db.squeeze(0), torch.tensor(label, dtype=torch.long)
