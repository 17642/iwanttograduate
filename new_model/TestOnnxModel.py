import os
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import onnxruntime as ort

# ── 설정 ──
ONNX_PATH   = "./model.onnx"   # ONNX 모델 파일 경로
TEST_DIR    = "./Sample/"             # 테스트할 .wav 파일들이 있는 디렉토리

SAMPLE_RATE = 41000
NUM_MELS    = 64
MAX_FRAMES  = 128

# ── 전처리 함수 ──
mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=NUM_MELS, hop_length=200, n_fft=400)
db_transform  = AmplitudeToDB()

def preprocess(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 정규화
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
    # Mel-DB 스펙트로그램
    mel = mel_transform(waveform)
    db  = db_transform(mel)
    spec = db.numpy().astype(np.float32)  # shape: (1, NUM_MELS, T)
    # 프레임 패딩/자르기
    t = spec.shape[2]
    if t < MAX_FRAMES:
        spec = np.pad(spec, ((0,0),(0,0),(0, MAX_FRAMES-t)))
    else:
        spec = spec[:, :, :MAX_FRAMES]
    # 배치 및 채널 차원 추가
    return spec[None, None, :, :]

# ── ONNX 세션 초기화 ──
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"]);
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"ONNX model loaded. Input='{input_name}', Output='{output_name}'")

# ── 테스트 루프 ──
def main():
    wavs = [f for f in os.listdir(TEST_DIR) if f.endswith('.wav')]
    if not wavs:
        print("No .wav files found for testing.")
        return

    correct = 0
    total   = 0

    for wav in sorted(wavs):
        # 파일명 앞 숫자를 실제 레이블로 사용 (1~15 → 0~14)
        try:
            true_label = int(wav.split('.',1)[0]) - 1
        except:
            print(f"Skipping '{wav}': cannot parse label.")
            continue

        x = preprocess(os.path.join(TEST_DIR, wav))
        # ONNX 추론
        out = sess.run([output_name], {input_name: x})[0].squeeze()
        pred = int(np.argmax(out))

        match = (pred == true_label)
        print(f"{wav:30} True: {true_label:2d}  Pred: {pred:2d}  Match: {match}")

        correct += int(match)
        total   += 1

    acc = correct/total*100 if total>0 else 0.0
    print(f"\nTest Completed: {total} files, Accuracy: {acc:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
