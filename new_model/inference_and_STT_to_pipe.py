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
import webrtcvad

from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

EVENT_PIPE = "/tmp/event_pipe"
STT_PIPE = "/tmp/stt_pipe"

HEF_PATH = "converted_model.hef"
NUM_MELS = 64
MAX_FRAMES = 128
SAMPLE_RATE = 44100
DURATION = 1.6   # 초
STT_LANGUAGE_CODE = "ko-KR"

DANGER_CLASS = set(range(14))

HEF_QUEUE = queue.Queue(maxsize=20)
STT_MODE_FLAG = threading.Event()   # 음성 감지 중에는 True

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
        normal_buffer = []
        normal_buffer_len = 0
        while True:
            # STT_MODE_FLAG가 켜지면(음성 감지) hef 추론 중단
            if STT_MODE_FLAG.is_set():
                time.sleep(0.1)
                continue
            # 평상시 1.6초 단위 버퍼
            indata = HEF_QUEUE.get()
            mel_db = preprocess_to_mel(indata)
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

def stt_google_streaming_thread():
    client = speech.SpeechClient()
    # 마이크에서 구글 STT로 음성 전달
    def stream_to_google():
        # 마이크로부터 직접 스트림 생성
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype='float32',
            blocksize=int(SAMPLE_RATE * 0.2),
        ) as stream:
            print("[STT] 음성 감지 대기중... (말하면 STT 시작)")
            while True:
                audio_frames = []
                STT_MODE_FLAG.wait()  # 누군가 음성 감지 → True로 set되면 STT시작
                print("[STT] 음성 감지됨! Google STT로 전송 시작")
                start_time = time.time()
                # 8초 제한 내에서 최대 10초(안 끊기면)까지
                while STT_MODE_FLAG.is_set():
                    frame, overflowed = stream.read(int(SAMPLE_RATE * 0.2))
                    audio_frames.append(frame[:,0])
                    # 10초 이상 연속 STT 금지(이상 감지용)
                    if time.time() - start_time > 10:
                        print("[STT] 10초 이상 음성, 강제 종료")
                        break
                # 음성 끝(플래그가 내려감)
                if audio_frames:
                    voice_data = np.concatenate(audio_frames)
                    yield voice_data
    def google_stt_once(voice_data):
        # voice_data: float32 numpy, range -1~1
        blocksize = 2048
        def gen():
            idx = 0
            while idx < len(voice_data):
                chunk = voice_data[idx:idx+blocksize]
                chunk_pcm = np.clip(chunk, -1, 1)
                chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                yield speech.StreamingRecognizeRequest(audio_content=chunk_pcm.tobytes())
                idx += blocksize
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=STT_LANGUAGE_CODE,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False,
            single_utterance=True
        )
        try:
            responses = client.streaming_recognize(streaming_config, gen())
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
            print(f"[STT] google_stt_once 내부 오류: {e}")

    # 실제 스레드 본체
    for voice_data in stream_to_google():
        if len(voice_data) > 0:
            google_stt_once(voice_data)
        else:
            print("[STT] 전달할 음성 데이터 없음 (voice_data 길이 0)")
        STT_MODE_FLAG.clear()  # STT 끝나면 hef 재개

def audio_input_thread():
    """
    항상 마이크에서 1.6초짜리 버퍼를 HEF_QUEUE에 밀어넣음.
    STT_MODE_FLAG가 켜지면 멈춤.
    webrtcvad로 실제 '말소리'만 감지해서 STT 전환.
    """
    vad = webrtcvad.Vad(2)  # 0~3: 높을수록 민감, 2 권장
    vad_sample_rate = 16000  # webrtcvad는 16kHz만 지원
    vad_frame_duration = 20  # ms (10, 20, 30만 가능)
    vad_frame_length = int(vad_sample_rate * vad_frame_duration / 1000)  # 320

    # 마이크 스트림은 여전히 44.1kHz float32로 받아오고, VAD 검사용 다운샘플 필요
    blocksize = int(SAMPLE_RATE * 0.2)
    normal_buffer = []
    normal_buffer_len = 0

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype='float32',
        blocksize=blocksize,
    ) as stream:
        while True:
            if STT_MODE_FLAG.is_set():
                time.sleep(0.1)
                continue
            frame, overflowed = stream.read(blocksize)
            frame = frame[:,0]

            # ----- webrtcvad용 다운샘플 및 변환 -----
            # 1. 44.1kHz float32 → 16kHz int16로 변환
            frame_resampled = np.interp(
                np.linspace(0, len(frame), int(len(frame) * vad_sample_rate / SAMPLE_RATE), endpoint=False),
                np.arange(len(frame)), frame
            ).astype(np.float32)
            frame_int16 = (np.clip(frame_resampled, -1, 1) * 32767).astype(np.int16)

            # 2. 20ms(320샘플)씩 쪼개서 VAD에 전달
            vad_detected = False
            for start in range(0, len(frame_int16) - vad_frame_length + 1, vad_frame_length):
                chunk = frame_int16[start:start+vad_frame_length].tobytes()
                if vad.is_speech(chunk, vad_sample_rate):
                    vad_detected = True
                    break

            # 실제 '말소리' 있을 때만 STT 모드 진입
            if vad_detected:
                STT_MODE_FLAG.set()
                normal_buffer = []
                normal_buffer_len = 0
                continue

            # --- 이하 기존 코드 (HEF용 1.6초 누적) ---
            normal_buffer.append(frame)
            normal_buffer_len += len(frame)
            if normal_buffer_len >= int(SAMPLE_RATE * DURATION):
                block = np.concatenate(normal_buffer)[:int(SAMPLE_RATE * DURATION)]
                try:
                    HEF_QUEUE.put(block, timeout=0.1)
                except queue.Full:
                    print("[HEF_QUEUE] 가득 참: 새 블록 버림!")
                remain = np.concatenate(normal_buffer)[int(SAMPLE_RATE * DURATION):]
                if len(remain) > 0:
                    normal_buffer = [remain]
                    normal_buffer_len = len(remain)
                else:
                    normal_buffer = []
                    normal_buffer_len = 0

def main():
    for pipe_path in [EVENT_PIPE, STT_PIPE]:
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
    threading.Thread(target=hef_inference_thread, daemon=True).start()
    threading.Thread(target=stt_google_streaming_thread, daemon=True).start()
    threading.Thread(target=audio_input_thread, daemon=True).start()
    try:
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
