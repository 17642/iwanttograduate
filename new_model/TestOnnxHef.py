import os
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import torch.nn.functional as F
import onnx

from hailo_platform import (
    HEF, VDevice,
    InferVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType
)
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ── 설정 ──
ONNX_PATH   = "./model.onnx"
HEF_PATH    = "./converted_model.hef"
TEST_DIR    = "./"  # 현재 폴더 내의 .wav 파일들

SAMPLE_RATE = 41000
NUM_MELS    = 64
MAX_FRAMES  = 128

# ── ONNX 모델 메타 확인 ──
print("Loading ONNX model metadata...")
onx = onnx.load(ONNX_PATH)
# 입력/출력 정보 출력
print("ONNX Inputs:")
for inp in onx.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")
print("ONNX Outputs:")
for out in onx.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {dims}")

# ── Hailo HEF 메타 확인 ──
print("\nInitializing Hailo device and loading HEF...")
device = VDevice()
hef = HEF(HEF_PATH)
ng = device.configure(hef)[0]
print("HEF Input Streams:")
for info in ng.get_input_vstream_infos():
    print(f"  {info.name}: shape={info.shape}, dtype={info.dtype}")
print("HEF Output Streams:")
for info in ng.get_output_vstream_infos():
    print(f"  {info.name}: shape={info.shape}, dtype={info.dtype}")

# ── 전처리 함수 ──
mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=NUM_MELS, hop_length=200, n_fft=400)
db_transform  = AmplitudeToDB()

def preprocess(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
    mel = mel_transform(waveform)
    db  = db_transform(mel)
    spec = db.numpy().astype(np.float32)
    t = spec.shape[2]
    if t < MAX_FRAMES:
        spec = np.pad(spec, ((0,0),(0,0),(0, MAX_FRAMES-t)))
    else:
        spec = spec[:, :, :MAX_FRAMES]
    return spec[None, None, :, :]

# ── 간단 테스트 수행 ──
def main():
    wavs = [f for f in os.listdir(TEST_DIR) if f.endswith('.wav')]
    print(f"Found {len(wavs)} WAV files for quick test.")
    # 랜덤 입력 테스트
    dummy = np.random.randn(1,1,NUM_MELS,MAX_FRAMES).astype(np.float32)
    with InferVStreams(ng,
                       InputVStreamParams.make(ng, format_type=FormatType.FLOAT32),
                       OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)) as pipeline:
        with ng.activate(ng.create_params()):
            res = pipeline.infer({ng.get_input_vstream_infos()[0].name: dummy})
    print("Dummy inference output shape:", res[ng.get_output_vstream_infos()[0].name].shape)

if __name__ == "__main__":
    main()
