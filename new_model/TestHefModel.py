import os
import numpy as np
import torch
torch.set_num_threads(1)  # 리소스 제한된 환경에서 안정적 실행
import torchaudio
import torch.nn.functional as F

from hailo_platform import (
    HEF, VDevice,
    InferVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType
)
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ── 설정 ──
MODEL_PATH = "converted_model.hef"
TEST_DIR   = "./"  # 현재 폴더 내의 .wav 파일들

SAMPLE_RATE = 16000
NUM_MELS    = 64
MAX_FRAMES  = 128

# ── Hailo 초기화 ──
print("Initializing Hailo device and network...")
device = VDevice()
hef = HEF(MODEL_PATH)
network_group = device.configure(hef)[0]
network_params = network_group.create_params()

input_info  = network_group.get_input_vstream_infos()[0]
output_info = network_group.get_output_vstream_infos()[0]

in_params  = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

# ── 전처리 함수 ──
mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=NUM_MELS, hop_length=200, n_fft=400)
db_transform  = AmplitudeToDB()

def preprocess(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    # 모노 변환
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 정규화
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

    # Mel-DB 스펙트로그램
    mel = mel_transform(waveform)
    db  = db_transform(mel)
    spec = db.numpy().astype(np.float32)

    # 프레임 패딩/자르기
    t = spec.shape[2]
    if t < MAX_FRAMES:
        spec = np.pad(spec, ((0,0),(0,0),(0, MAX_FRAMES-t)))
    else:
        spec = spec[:, :, :MAX_FRAMES]

    return spec[None, None, :, :]

# ── 메인 테스트 루프 ──
def main():
    wavs = [f for f in os.listdir(TEST_DIR) if f.endswith('.wav')]
    if not wavs:
        print("No .wav files found in", TEST_DIR)
        return

    correct = 0
    total = len(wavs)
    print(f"Found {total} test files. Running inference...")

    with InferVStreams(network_group, in_params, out_params) as infer_pipeline:
        for wav in wavs:
            path = os.path.join(TEST_DIR, wav)
            # 파일명에서 true label 추출 (예: '5.화재_...')
            try:
                true_label = int(wav.split('.', 1)[0])
            except ValueError:
                true_label = None

            x = preprocess(path)
            with network_group.activate(network_params):
                results = infer_pipeline.infer({input_info.name: x})
            out = results[output_info.name][0]
            pred = int(np.argmax(out))

            match = (pred == true_label)
            if match:
                correct += 1

            print(f"{wav} -> True: {true_label}, Pred: {pred}, Scores: {out.tolist()}, Correct: {match}")

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
