import time
import queue
import threading
import paho.mqtt.client as mqtt
import sounddevice as sd
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import speech_recognition as sr
# Hailo HEF 라이브러리 임포트 (실제 환경에서 경로와 import 방식 확인)

from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

# 1. MQTT AWS 설정 =========================
MQTT_BROKER = "AWS_IOT_CORE_ENDPOINT"   # 예: "abcdefgh.iot.ap-northeast-2.amazonaws.com"
MQTT_PORT = 8883
CA_PATH = "AmazonRootCA1.pem"
CERT_PATH = "device-certificate.pem.crt"
KEY_PATH = "private.pem.key"
MQTT_TOPIC = "home/sensors/hazard"

# 2. 오디오/모델/전처리 파라미터 ===========
HEF_PATH = "converted_model.hef"
NUM_MELS = 64
MAX_FRAMES = 128
SAMPLE_RATE = 16000
DURATION = 1.0   # 초

# 3. 위험 클래스 목록 ======================
DANGER_CLASS = set(range(14))  # 0~13이 위험, 14는 안전

# 4. 공유 오디오 큐 =======================
AUDIO_QUEUE = queue.Queue()

# 5. Mel-spectrogram 전처리 함수 ==========
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
    # (1, 64, 128) → (1, 1, 64, 128)
    return spec[:, None, :, :]

# 6. MQTT Client 생성 (AWS IoT Core) =======
def make_aws_mqtt_client():
    client = mqtt.Client()
    client.tls_set(ca_certs=CA_PATH,
                   certfile=CERT_PATH,
                   keyfile=KEY_PATH)
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()
    return client

aws_mqtt = make_aws_mqtt_client()

# 7. 위험 상황 MQTT 전송 함수 ===============
def send_hazard_mqtt(pred, scores):
    import json
    msg = {
        "timestamp": time.time(),
        "type": "audio_hazard",
        "pred": int(pred),
        "scores": [float(x) for x in scores]
    }
    aws_mqtt.publish(MQTT_TOPIC, json.dumps(msg))

def send_stt_mqtt(text):
    import json
    msg = {
        "timestamp": time.time(),
        "type": "stt",
        "text": text
    }
    aws_mqtt.publish(MQTT_TOPIC, json.dumps(msg))

# 8. HEF 추론 (별도 쓰레드) ================
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
            # 전처리
            mel_db = preprocess_to_mel(audio)  # (1,1,64,128)
            input_data = {input_info.name: mel_db}
            infer_results = infer_pipeline.infer(input_data)
            scores = infer_results[output_info.name][0]
            pred = int(np.argmax(scores))
            print(f"[HEF 추론] pred={pred}, scores={scores}")
            # 위험일 때만 전송
            if pred in DANGER_CLASS:
                send_hazard_mqtt(pred, scores)
            AUDIO_QUEUE.task_done()

# 9. STT 쓰레드 ===========================
def stt_thread():
    recognizer = sr.Recognizer()
    while True:
        audio = AUDIO_QUEUE.get()
        # sr.AudioData expects int16, 2 bytes per sample
        audio16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio16.tobytes()
        audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        try:
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            print(f"[STT 결과] {text}")
            send_stt_mqtt(text)
        except sr.UnknownValueError:
            print("[STT] 인식 불가")
        except Exception as e:
            print("[STT] 오류:", e)
        AUDIO_QUEUE.task_done()

# 10. 오디오 콜백 =========================
def audio_callback(indata, frames, time_info, status):
    # float32, [-1.0, 1.0], shape=(blocksize, 1)
    wave = indata[:,0].copy()
    AUDIO_QUEUE.put(wave)

# 11. 메인 ================================
def main():
    # 쓰레드 2개 (HEF, STT)
    threading.Thread(target=hef_inference_thread, daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()
    # 실시간 오디오 입력
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
