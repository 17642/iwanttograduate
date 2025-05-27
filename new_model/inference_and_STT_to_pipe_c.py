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
import errno

from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

# ====== 파이프 경로 ======
EVENT_PIPE = "/tmp/event_pipe"   # 모델 추론 + 레이더 결과
STT_PIPE = "/tmp/stt_pipe"       # STT 변환 결과

# ====== 오디오/모델 파라미터 ======
HEF_PATH = "converted_model.hef"
NUM_MELS = 64
MAX_FRAMES = 128
SAMPLE_RATE = 44100
HEF_DURATION = 1.6     # HEF 추론용
STT_DURATION = 8.0     # STT 변환용

DANGER_CLASS = set(range(14))  # 0~13이 위험, 14는 안전

# === 큐: 위험감지/음성인식 각각 독립 큐 사용 ===
HEF_QUEUE = queue.Queue(maxsize=20)
STT_QUEUE = queue.Queue(maxsize=5)

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=NUM_MELS,
    hop_length=200,
    n_fft=400
)
db_transform = AmplitudeToDB()

STT_LANGUAGE_CODE = "ko-KR"

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
    spec = np.transpose(spec, (1,2,0))  # (64, 128, 1)
    return spec

def send_event_pipe(msg_dict):
    try:
        with open(EVENT_PIPE, "w") as f:
            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print("[PIPE] 기록 실패:", e)

def send_stt_pipe(msg_dict):
    try:
        with open(STT_PIPE, "w") as f:
            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print("[STT_PIPE] 기록 실패:", e)

def hef_inference_thread():
    device = VDevice()
    hef = HEF(HEF_PATH)
    network_group = device.configure(hef)[0]
    input_info = network_group.get_input_vstream_infos()[0]
    output_info = network_group.get_output_vstream_infos()[0]
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while True:
            audio = HEF_QUEUE.get()
            mel_db = preprocess_to_mel(audio)
            mel_db = np.expand_dims(mel_db,axis=0)
            input_data = {input_info.name: mel_db}
            with network_group.activate():
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
            HEF_QUEUE.task_done()

def stt_thread():
    client = speech.SpeechClient()
    while True:
        audio = STT_QUEUE.get()
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        audio_content = audio_int16.tobytes()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=STT_LANGUAGE_CODE,
            enable_automatic_punctuation=True,
        )
        audio_data = speech.RecognitionAudio(content=audio_content)
        try:
            response = client.recognize(config=config, audio=audio_data)
            text = ""
            for result in response.results:
                if result.alternatives:
                    text += result.alternatives[0].transcript
            print(f"[STT 결과] {text}")
            msg = {
                "timestamp": time.time(),
                "type": "stt",
                "text": text if text else ""
            }
            send_stt_pipe(msg)
        except Exception as e:
            print("[STT] 오류:", e)
        STT_QUEUE.task_done()

# ---- 오디오 콜백에서 두 rolling buffer 관리 ----
hef_buffer = []
hef_buffer_len = 0
stt_buffer = []
stt_buffer_len = 0
buffer_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    global hef_buffer, hef_buffer_len, stt_buffer, stt_buffer_len
    wave = indata[:,0].copy()

    with buffer_lock:
        # HEF용 1.6초 rolling
        hef_buffer.append(wave)
        hef_buffer_len += len(wave)
        if hef_buffer_len >= int(SAMPLE_RATE * HEF_DURATION):
            block = np.concatenate(hef_buffer)[:int(SAMPLE_RATE * HEF_DURATION)]
            try:
                HEF_QUEUE.put(block, timeout=0.1)
            except queue.Full:
                print("[HEF_QUEUE] 가득 참: 새 블록 버림!")
            remain = np.concatenate(hef_buffer)[int(SAMPLE_RATE * HEF_DURATION):]
            if len(remain) > 0:
                hef_buffer = [remain]
                hef_buffer_len = len(remain)
            else:
                hef_buffer = []
                hef_buffer_len = 0

        # STT용 8초 rolling
        stt_buffer.append(wave)
        stt_buffer_len += len(wave)
        if stt_buffer_len >= int(SAMPLE_RATE * STT_DURATION):
            stt_block = np.concatenate(stt_buffer)[:int(SAMPLE_RATE * STT_DURATION)]
            try:
                STT_QUEUE.put(stt_block, timeout=0.1)
            except queue.Full:
                print("[STT_QUEUE] 가득 참: 새 STT 블록 버림!")
            remain = np.concatenate(stt_buffer)[int(SAMPLE_RATE * STT_DURATION):]
            if len(remain) > 0:
                stt_buffer = [remain]
                stt_buffer_len = len(remain)
            else:
                stt_buffer = []
                stt_buffer_len = 0

def main():
    for pipe_path in [EVENT_PIPE, STT_PIPE]:
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
    threading.Thread(target=hef_inference_thread, daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()
    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE*0.2),
            callback=audio_callback,
        ):
            print("실시간 감지 시작 (Ctrl+C로 종료)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("프로그램 종료 요청(Ctrl+C) 감지!")
    finally:
        print("InputStream/리소스 정리 완료, 안전하게 종료.")
        for pipe_path in [EVENT_PIPE, STT_PIPE]:
            try:
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
                    print(f"파이프 파일 {pipe_path} 삭제 완료.")
            except OSError as e:
                if e.errno != errno.ENOENT:
                    print(f"파이프 파일 삭제 중 오류({pipe_path}): {e}")

if __name__ == "__main__":
    main()
