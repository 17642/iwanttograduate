import time
import queue
import threading
import sounddevice as sd
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import speech_recognition as sr
import json
import os

# Hailo HEF 라이브러리 (환경에 맞게 경로/이름 조정!)
from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

# ====== 파이프 경로 ======
HAZARD_PIPE = "/tmp/hazard_pipe"
STT_PIPE = "/tmp/stt_pipe"

# ====== 오디오/모델 파라미터 ======
HEF_PATH = "converted_model.hef"
NUM_MELS = 64
MAX_FRAMES = 128
SAMPLE_RATE = 16000
DURATION = 1.0   # 초

DANGER_CLASS = set(range(14))  # 0~13이 위험, 14는 안전

AUDIO_QUEUE = queue.Queue()

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=NUM_MELS,
    hop_length=200,
    n_fft=400
)
db_transform = AmplitudeToDB()

def preprocess_to_mel(wave):
    tensor = torch.from_numpy(wave).unsqueeze(0)
    mel = mel_transform(tensor)
    db = db_transform(mel)
    spec = db.numpy().astype(np.float32)
    t = spec.shape[2]
    if t < MAX_FRAMES:
        spec = np.pad(spec, ((0,0), (0,0), (0, MAX_FRAMES-t)))
    else:
        spec = spec[:, :, :MAX_FRAMES]
    return spec[:, None, :, :]  # (1, 1, 64, 128)

# --- HEF 추론 결과를 파이프에 기록 ---
def send_hazard_pipe(pred, scores):
    msg = {
        "timestamp": time.time(),
        "type": "audio_hazard",
        "pred": int(pred),
        "scores": [float(x) for x in scores]
    }
    with open(HAZARD_PIPE, "w") as f:
        f.write(json.dumps(msg) + "\n")

# --- STT 결과를 파이프에 기록 ---
def send_stt_pipe(text):
    msg = {
        "timestamp": time.time(),
        "type": "stt",
        "text": text
    }
    with open(STT_PIPE, "w") as f:
        f.write(json.dumps(msg) + "\n")

def hef_inference_thread():
    device = VDevice()
    hef = HEF(HEF_PATH)
    network_group = device.configure(hef)[0]
    network_group.activate()
    input_info = network_group.get_input_vstream_infos()[0]
    output_info = network_group.get_output_vstream_infos()[0]
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while True:
            audio = AUDIO_QUEUE.get()
            mel_db = preprocess_to_mel(audio)  # (1,1,64,128)
            input_data = {input_info.name: mel_db}
            infer_results = infer_pipeline.infer(input_data)
            scores = infer_results[output_info.name][0]
            pred = int(np.argmax(scores))
            print(f"[HEF 추론] pred={pred}, scores={scores}")
            if pred in DANGER_CLASS:
                send_hazard_pipe(pred, scores)
            AUDIO_QUEUE.task_done()

def stt_thread():
    recognizer = sr.Recognizer()
    while True:
        audio = AUDIO_QUEUE.get()
        audio16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio16.tobytes()
        audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        try:
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            print(f"[STT 결과] {text}")
            send_stt_pipe(text)
        except sr.UnknownValueError:
            print("[STT] 인식 불가")
        except Exception as e:
            print("[STT] 오류:", e)
        AUDIO_QUEUE.task_done()

def audio_callback(indata, frames, time_info, status):
    wave = indata[:,0].copy()
    AUDIO_QUEUE.put(wave)

def main():
    # 파이프 없으면 생성
    for pipe_path in [HAZARD_PIPE, STT_PIPE]:
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
    # HEF, STT 쓰레드 실행
    threading.Thread(target=hef_inference_thread, daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()
    blocksize = int(SAMPLE_RATE * DURATION)
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=blocksize,
        callback=audio_callback,
    ):
        print("실시간 감지 시작 (Ctrl+C로 종료)")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
