import time
import queue
import threading
import sounddevice as sd
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from google.cloud import speech
import json
import os
import wave

from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

# ====== 파이프 경로 ======
EVENT_PIPE = "/tmp/event_pipe"   # 모델 추론 + 레이더 결과
STT_PIPE = "/tmp/stt_pipe"       # STT 변환 결과

# ====== 오디오/모델 파라미터 ======
HEF_PATH = "converted_model.hef"
NUM_MELS = 64
MAX_FRAMES = 128
SAMPLE_RATE = 44100
DURATION = 1.6   # 초

DANGER_CLASS = set(range(14))  # 0~13이 위험, 14는 안전

AUDIO_QUEUE = queue.Queue()

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=NUM_MELS,
    hop_length=200,
    n_fft=400
)
db_transform = AmplitudeToDB()

STT_LANGUAGE_CODE = "ko-KR"
STT_SAMPLE_RATE = 16000

def wave_to_temp_wavfile(wave_np, filename, samplerate=16000):
    # wave_np: float32, [-1, 1]
    audio_int16 = (wave_np * 32767).astype(np.int16)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())
    return filename

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

# --- event_pipe에 메시지 기록 ---
def send_event_pipe(msg_dict):
    try:
        with open(EVENT_PIPE, "w") as f:
            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[PIPE] 기록 실패:", e)

# --- stt_pipe에 메시지 기록 ---
def send_stt_pipe(msg_dict):
    try:
        with open(STT_PIPE, "w") as f:
            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[STT_PIPE] 기록 실패:", e)

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
            mel_db = preprocess_to_mel(audio)
            input_data = {input_info.name: mel_db}
            infer_results = infer_pipeline.infer(input_data)
            scores = infer_results[output_info.name][0]
            pred = int(np.argmax(scores))
            print(f"[HEF 추론] pred={pred}, scores={scores}")
            if pred in DANGER_CLASS:
                msg = {
                    "timestamp": time.time(),
                    "type": "audio_hazard",
                    "pred": pred,
                    "scores": [float(x) for x in scores]
                }
                send_event_pipe(msg)
            AUDIO_QUEUE.task_done()

def stt_thread():
    client = speech.SpeechClient()
    while True:
        audio = AUDIO_QUEUE.get()
        temp_path = "/tmp/stt_temp.wav"
        wave_to_temp_wavfile(audio, temp_path, samplerate=STT_SAMPLE_RATE)
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        audio_proto = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=STT_LANGUAGE_CODE,
            enable_automatic_punctuation=True,
        )
        try:
            response = client.recognize(config=config, audio=audio_proto)
            text = ""
            for result in response.results:
                text = result.alternatives[0].transcript
                break
            print(f"[STT 결과] {text}")
            msg = {
                "timestamp": time.time(),
                "type": "stt",
                "text": text if text else ""
            }
            send_stt_pipe(msg)
        except Exception as e:
            print("[STT] 오류:", e)
        AUDIO_QUEUE.task_done()

def audio_callback(indata, frames, time_info, status):
    wave = indata[:,0].copy()
    AUDIO_QUEUE.put(wave)

def main():
    # 파이프 없으면 생성
    for pipe_path in [EVENT_PIPE, STT_PIPE]:
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
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
