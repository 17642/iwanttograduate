import os
import paho.mqtt.client as mqtt
import json
import time

EVENT_PIPE = "/tmp/event_pipe"
if not os.path.exists(EVENT_PIPE):
    os.mkfifo(EVENT_PIPE)

MQTT_BROKER = "localhost"   # 또는 ESP32에서 전송하는 브로커의 IP
MQTT_PORT = 1883
MQTT_TOPIC = "home/sensors/fall"

def send_event_pipe(msg_dict):
    try:
        with open(EVENT_PIPE, "w") as f:
            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[PIPE] 기록 실패:", e)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        # ESP32에서 이미 JSON이면 pass, 아니라도 강제 변환
        data = json.loads(payload)
        data["timestamp"] = time.time()  # 서버 수신 시각
        data["type"] = "radar"
        send_event_pipe(data)
    except Exception as e:
        print("RADAR_PIPE 기록 실패:", e)

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(MQTT_TOPIC)
client.loop_forever()
