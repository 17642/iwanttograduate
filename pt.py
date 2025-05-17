import numpy as np
import sounddevice as sd
import onnxruntime as ort
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ── 설정 ──
HEF_PATH      = "./cnn_hazard_detector.hef"
ONNX_PATH     = "./cnn_hazard_detector.onnx"  # 세션 생성 시 필요
SAMPLE_RATE   = 16000
DURATION      = 1.0
NUM_MELS      = 64
HOP_LENGTH    = 200
MAX_FRAMES    = 128

# ── ONNX Runtime 세션 설정 ──
sess_opts = ort.SessionOptions()
# Hailo EP용 커스텀 OPS 라이브러리 로드 (환경에 맞게 경로 조정)
sess_opts.register_custom_ops_library("libhailort.so")
# HEF를 명시적으로 지정
sess_opts.add_session_config_entry("HAILO_HEF_PATH", HEF_PATH)

# EP 우선순위: Hailo → CPU
providers = ['HailoExecutionProvider', 'CPUExecutionProvider']
session   = ort.InferenceSession(ONNX_PATH, sess_options=sess_opts, providers=providers)
print("Active providers:", session.get_providers())

input_name = session.get_inputs()[0].name

# ── Mel-Spectrogram 전처리 ──
mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE,
                               n_mels=NUM_MELS,
                               hop_length=HOP_LENGTH,
                               n_fft=HOP_LENGTH*2)
db_transform  = AmplitudeToDB()

def preprocess(wave: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(wave).unsqueeze(0)        # (1, samples)
    mel    = mel_transform(tensor)                     # (1, n_mels, time)
    db     = db_transform(mel)                         # (1, n_mels, time)
    spec   = db.numpy().astype(np.float32)             # numpy array

    # pad/truncate to MAX_FRAMES
    t = spec.shape[2]
    if t < MAX_FRAMES:
        spec = np.pad(spec, ((0,0),(0,0),(0, MAX_FRAMES-t)))
    else:
        spec = spec[:,:,:MAX_FRAMES]
        
    spec = spec[:, None, :, :]   # axis=1에 채널 축(1) 추가

    return spec          # (1,1,n_mels,MAX_FRAMES)

# ── 실시간 오디오 스트림 & 추론 ──
def callback(indata, frames, time, status):
    if status:
        print("Status:", status)
    audio = indata[:,0]
    x = preprocess(audio)
    out = session.run(None, {input_name: x})[0].squeeze()
    pred = int(np.argmax(out))
    print(f"Pred: {pred}  Scores: {out.tolist()}")

def main():
    block = int(SAMPLE_RATE * DURATION)
    with sd.InputStream(channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=block,
                        callback=callback):
        print(">>> Listening. Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped.")

if __name__ == "__main__":
    main()
