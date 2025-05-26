import numpy as np
import sounddevice as sd
import queue
import threading
import time
import paho.mqtt.client as mqtt
# Hailo HEF 추론용 라이브러리 (실제 환경에 맞게 import 필요)
from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType
# Google STT용
import speech_recognition as sr

# ───── MQTT 설정 ─────
MQTT_BROKER = '서버_IP'
MQTT_PORT   = 1883
MQTT_TOPIC  = 'home/sensors/hazard'

# ───── Hailo HEF 설정 ─────
HEF_PATH = "converted_model.hef"
NUM_MELS = 64
MAX_FRAMES = 128
SAMPLE_RATE = 16000
DURATION = 1.0  # 1초 단위 추론

# ───── 음성 버퍼 (공유 큐) ─────
AUDIO_QUEUE = queue.Queue()

def mqtt_send(label):
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    payload = {"timestamp": time.time(), "hazard": label}
    client.publish(MQTT_TOPIC, str(payload))
    client.disconnect()

# ───── HEF 추론 쓰레드 ─────
def hef_inference_thread():
    # Hailo 환경 초기화
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
            # ------ 전처리 (STFT, Mel-Spectrogram 등) ------
            # audio: (샘플수,)  float32 np.ndarray
            # 1초(16000 samples) 입력이라고 가정
            # (실제 Mel-Spectrogram 전처리 코드는 기존 코드 활용)
            mel_db = preprocess_to_mel(audio)  # (1,64,128,1) float32
            
            input_data = {input_info.name: mel_db}
            infer_results = infer_pipeline.infer(input_data)
            scores = infer_results[output_info.name][0]
            pred = int(np.argmax(scores))
            print(f"[HEF 추론] pred={pred} scores={scores}")
            
            if pred in 위험클래스목록:  # 0~13이면 위험
                mqtt_send(pred)
            AUDIO_QUEUE.task_done()

# ───── STT 쓰레드 ─────
def stt_thread():
    recognizer = sr.Recognizer()
    while True:
        audio = AUDIO_QUEUE.get()
        audio_data = sr.AudioData(audio.tobytes(), SAMPLE_RATE, 2)
        try:
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            print(f"[STT 결과] {text}")
            # 필요시 추가 처리
        except sr.UnknownValueError:
            print("[STT] 인식 불가")
        except Exception as e:
            print("[STT] 오류:", e)
        AUDIO_QUEUE.task_done()

# ───── 오디오 캡처 콜백 ─────
def audio_callback(indata, frames, time_info, status):
    # indata: (blocksize, 1) float32
    AUDIO_QUEUE.put(indata[:, 0].copy())  # 1D float32 (SAMPLE_RATE samples)

def main():
    # 멀티쓰레드 실행
    threading.Thread(target=hef_inference_thread, daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()
    
    # 실시간 오디오 스트림
    blocksize = int(SAMPLE_RATE * DURATION)
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=blocksize,
        callback=audio_callback,
    ):
        print("실시간 음성 감지 시작 (Ctrl+C 종료)")
        while True:
            time.sleep(1)

# ---------- Mel-Spectrogram 전처리 함수 (기존 코드 활용) ----------
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=NUM_MELS, hop_length=200, n_fft=400)
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
    spec = np.transpose(spec, (0,2,1))  # (1,128,64) → 필요시 (1,64,128,1)
    spec = spec[:, None, :, :]  # (1,1,64,128)
    return spec

# ---------- 위험 클래스 목록 ----------
위험클래스목록 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 14번이 '안전'

if __name__ == "__main__":
    main()
