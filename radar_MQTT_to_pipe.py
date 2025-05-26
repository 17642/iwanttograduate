import os
import paho.mqtt.client as mqtt
import json

RADAR_PIPE = "/tmp/radar_pipe"
if not os.path.exists(RADAR_PIPE):
    os.mkfifo(RADAR_PIPE)

MQTT_BROKER = "localhost"   # 또는 ESP32에서 전송하는 브로커의 IP
MQTT_PORT = 1883
MQTT_TOPIC = "home/sensors/fall"

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        # JSON 포맷이면 그대로 사용
        data = json.loads(payload)
        with open(RADAR_PIPE, "w") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print("RADAR_PIPE 기록 실패:", e)

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(MQTT_TOPIC)
client.loop_forever()
