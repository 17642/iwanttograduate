import paho.mqtt.client as mqtt
import ssl

# AWS IoT Core 세팅
AWS_ENDPOINT = "aabbccddee-ats.iot.ap-northeast-2.amazonaws.com"
AWS_PORT = 8883
CA_PATH = "AmazonRootCA1.pem"
CERT_PATH = "device-certificate.pem.crt"
KEY_PATH = "private.pem.key"
AWS_TOPIC = "home/sensors/fall"

# 1. 내부 MQTT 브로커(라즈베리파이 Mosquitto)에 subscribe
LOCAL_BROKER = "localhost"
LOCAL_PORT = 1883
LOCAL_TOPIC = "home/sensors/fall"

def on_local_message(client, userdata, msg):
    print(f"수신: {msg.payload}")
    # AWS로 publish
    aws_client.publish(AWS_TOPIC, msg.payload)

# 2. AWS MQTT Client 세팅
aws_client = mqtt.Client()
aws_client.tls_set(ca_certs=CA_PATH,
                   certfile=CERT_PATH,
                   keyfile=KEY_PATH,
                   tls_version=ssl.PROTOCOL_TLSv1_2)
aws_client.connect(AWS_ENDPOINT, AWS_PORT)

# 3. 로컬 MQTT Client 세팅 (subscribe)
local_client = mqtt.Client()
local_client.on_message = on_local_message
local_client.connect(LOCAL_BROKER, LOCAL_PORT)
local_client.subscribe(LOCAL_TOPIC)
local_client.loop_start()

# AWS도 연결 (keepalive)
aws_client.loop_start()

import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("종료")

aws_client.loop_stop()
local_client.loop_stop()
