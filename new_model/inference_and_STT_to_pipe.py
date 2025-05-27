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

# === 큐: 위험감지/음성인식 각각 독립 큐 사용 ===
HEF_QUEUE = queue.Queue()
STT_QUEUE = queue.Queue()

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=NUM_MELS,
    hop_length=200,
    n_fft=400
)
db_transform = AmplitudeToDB()

VAD_ENERGY_THRESHOLD = 0.02   # 실험적으로 조절 필요
VAD_MIN_DURATION = 0.4        # 음성이 0.4초 이상 지속되어야 "음성시작" 인정
STT_LANGUAGE_CODE = "ko-KR"

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
        
    spec = np.transpose(spec, (1,2,0))  # (64, 128, 1)
    return spec

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
    input_info = network_group.get_input_vstream_infos()[0]
    output_info = network_group.get_output_vstream_infos()[0]
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    #input_vstreams_params = InputVStreamParams.make(name=input_info.name, shape=input_info.shape, format_type=FormatType.FLOAT32,quant_info = input_info.quant_info, timeout_ms=1000,  frames_count=1)
    #input_vstreams_params = [input_vstreams_params]

    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while True:
            audio = HEF_QUEUE.get()
            mel_db = preprocess_to_mel(audio)
            mel_db = np.expand_dims(mel_db,axis=0)
            #new_input_data = np.zeros((64,64,128,1)).astype(np.float32)
            input_data = {input_info.name: mel_db}
            #print(input_info)
            #print("Expected input shape:", input_info.shape)
            #print("Actual input shape:", mel_db.shape)
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
        # 1. 실시간 스트리밍 방식으로 변환
        def audio_generator():
            blocksize = 2048
            idx = 0
            while idx < len(audio):
                chunk = audio[idx:idx+blocksize]
                yield speech.StreamingRecognizeRequest(audio_content=chunk.astype(np.float32).tobytes())
                idx += blocksize

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=STT_LANGUAGE_CODE,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False,
            single_utterance=True
        )
        try:
            responses = client.streaming_recognize(streaming_config, audio_generator())
            text = ""
            for response in responses:
                for result in response.results:
                    if result.alternatives:
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
        STT_QUEUE.task_done()

# 전역 변수
normal_buffer = []
normal_buffer_len = 0

def audio_callback(indata, frames, time_info, status):
    global vad_voice_active, vad_voice_buffer, vad_voice_frames
    global normal_buffer, normal_buffer_len

    wave = indata[:,0].copy()
    energy = np.sqrt(np.mean(wave**2))

    if not vad_voice_active:
        # === 평상시: 1.6초 블록 쌓기 ===
        normal_buffer.append(wave)
        normal_buffer_len += len(wave)
        # VAD 체크
        if energy > VAD_ENERGY_THRESHOLD:
            vad_voice_frames += 1
            vad_voice_buffer.append(wave)
            if vad_voice_frames * frames/SAMPLE_RATE >= VAD_MIN_DURATION:
                vad_voice_active = True
                print("[VAD] 음성 감지 시작!")
                # 음성 시작하면 평상시 버퍼 비우기(누적 중단)
                normal_buffer = []
                normal_buffer_len = 0
        else:
            vad_voice_frames = 0
            vad_voice_buffer.clear()
        # 1.6초 쌓이면 HEF_QUEUE로
        if normal_buffer_len >= int(SAMPLE_RATE * DURATION):
            # 초과분은 잘라서 처리
            buffer_concat = np.concatenate(normal_buffer)
            hef_block = buffer_concat[:int(SAMPLE_RATE * DURATION)]
            HEF_QUEUE.put(hef_block)
            # 남은 샘플은 다음 블록으로(1.6초 단위로 rolling)
            remain = buffer_concat[int(SAMPLE_RATE * DURATION):]
            normal_buffer = [remain] if len(remain) > 0 else []
            normal_buffer_len = len(remain)
    else:
        vad_voice_buffer.append(wave)
        if energy > VAD_ENERGY_THRESHOLD:
            vad_voice_frames = 0
        else:
            vad_voice_frames += 1
            if vad_voice_frames * frames/SAMPLE_RATE >= 0.5:
                voice_chunk = np.concatenate(vad_voice_buffer)
                STT_QUEUE.put(voice_chunk)
                vad_voice_active = False
                vad_voice_frames = 0
                vad_voice_buffer.clear()
                print("[VAD] 음성 종료, STT 전송")
    

def main():
    global vad_voice_active, vad_voice_buffer, vad_voice_frames
    vad_voice_active = False
    vad_voice_buffer = []
    vad_voice_frames = 0

    for pipe_path in [EVENT_PIPE, STT_PIPE]:
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
    threading.Thread(target=hef_inference_thread, daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()

    # 0.2초 단위로 프레임 전송(VAD 반응성↑)
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE*0.2),
        callback=audio_callback,
    ):
        print("실시간 감지 시작 (Ctrl+C로 종료)")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
