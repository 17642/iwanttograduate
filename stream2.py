#!/usr/bin/env python3

import argparse
from math import sqrt
import cv2
from pose_utils import postproc_yolov8_pose
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo
from collections import deque
import boto3
import RPi.GPIO as GPIO
import time
import threading
import subprocess

parser = argparse.ArgumentParser(description='Pose estimation and streaming with Hailo and Kinesis')
parser.add_argument('-m', '--model', help="HEF file path", default="/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef")
args = parser.parse_args()

# 키포인트 인덱스 정의
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

JOINT_PAIRS = [[NOSE, L_EYE], [L_EYE, L_EAR], [NOSE, R_EYE], [R_EYE, R_EAR],
               [L_SHOULDER, R_SHOULDER],
               [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST],
               [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP],
               [L_HIP, L_KNEE], [R_HIP, R_KNEE], [L_KNEE, L_ANKLE], [R_KNEE, R_ANKLE]]

# 설정
hip_y_buffer = deque(maxlen=15)
sns_client = boto3.client('sns', region_name='us-east-1')
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:FallDetectionTopic'
BUTTON_PIN = 17
BUZZER_PIN = 18
USE_BUZZER = True

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
buzzer = None
if USE_BUZZER:
    try:
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        buzzer = GPIO.PWM(BUZZER_PIN, 1000)
        print("Buzzer initialized")
    except Exception as e:
        print(f"Failed to initialize buzzer: {e}")
        USE_BUZZER = False

# 상태 변수
fall_detected = False
button_pressed = False
last_fall_time = 0

# GStreamer 스트리밍 파이프라인
STREAM_NAME = "MyStream"
REGION = "us-east-1"
gstreamer_pipeline = (
    f"rpicamsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 ! h264parse ! "
    f"kvssink stream-name={STREAM_NAME} aws-region={REGION}"
)
streaming_process = None

def start_streaming():
    global streaming_process
    try:
        streaming_process = subprocess.Popen(["gst-launch-1.0", "-v"] + gstreamer_pipeline.split())
        print("Kinesis Video Streams streaming started")
    except Exception as e:
        print(f"Failed to start streaming: {e}")

def stop_streaming():
    if streaming_process:
        streaming_process.terminate()
        print("Kinesis Video Streams streaming stopped")

def is_fallen(results, hip_y_buffer, model_size, detection_threshold=0.5, joint_threshold=0.5):
    bboxes, scores, keypoints, joint_scores = (
        results['bboxes'], results['scores'], results['keypoints'], results['joint_scores'])
    box, score, keypoint, keypoint_score = bboxes[0], scores[0], keypoints[0], joint_scores[0]

    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in (
            zip(box, score, keypoint, keypoint_score)):
        if detection_score < detection_threshold:
            continue

        joint_visible = detection_keypoints_score > joint_threshold
        detection_keypoints = detection_keypoints.reshape(17, 2)

        left_shoulder = detection_keypoints[L_SHOULDER]
        right_shoulder = detection_keypoints[R_SHOULDER]
        left_hip = detection_keypoints[L_HIP]
        right_hip = detection_keypoints[R_HIP]

        if (joint_visible[L_SHOULDER] and joint_visible[R_SHOULDER] and
            joint_visible[L_HIP] and joint_visible[R_HIP]):

            curr_hip_y = (left_hip[1] + right_hip[1]) / 2
            hip_y_buffer.append(curr_hip_y)

            hip_drop = False
            if len(hip_y_buffer) == hip_y_buffer.maxlen:
                oldest_hip_y = hip_y_buffer[0]
                hip_drop_speed = curr_hip_y - oldest_hip_y
                if hip_drop_speed > 50:
                    hip_drop = True

            if hip_drop:
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                shoulder_twist = shoulder_diff > 30
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_y = (left_hip[1] + right_hip[1]) / 2
                height_diff = abs(shoulder_y - hip_y)
                similar_height = height_diff < 100

                if shoulder_twist or similar_height:
                    return True
    return False

def play_alert_sound():
    if USE_BUZZER and buzzer:
        buzzer.start(50)
        print("Buzzer alert started")

def stop_alert_sound():
    if USE_BUZZER and buzzer:
        buzzer.stop()
        print("Buzzer alert stopped")

def send_sns_notification():
    message = f"Emergency: Fall detected at {time.strftime('%Y-%m-%d %H:%M:%S')} and no response within 20 seconds!"
    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=message,
            Subject="Emergency Fall Alert"
        )
        print("SNS emergency notification sent to guardian")
    except Exception as e:
        print(f"Failed to send SNS notification: {e}")

def handle_fall_alert():
    global button_pressed, fall_detected
    play_alert_sound()
    start_time = time.time()

    while time.time() - start_time < 20:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            button_pressed = True
            stop_alert_sound()
            print("Button pressed, fall confirmed as safe")
            break
        time.sleep(0.1)

    if not button_pressed:
        stop_alert_sound()
        send_sns_notification()
    fall_detected = False

def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.5, joint_threshold=0.5):
    image_size = (image.shape[1], image.shape[0])

    def scale_coord(coord):
        return tuple([int(c * t / f) for c, f, t in zip(coord, model_size, image_size)])

    bboxes, scores, keypoints, joint_scores = (
        results['bboxes'], results['scores'], results['keypoints'], results['joint_scores'])
    box, score, keypoint, keypoint_score = bboxes[0], scores[0], keypoints[0], joint_scores[0]

    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in (
            zip(box, score, keypoint, keypoint_score)):
        if detection_score < detection_threshold:
            continue

        coord_min = scale_coord(detection_box[:2])
        coord_max = scale_coord(detection_box[2:])
        cv2.rectangle(image, coord_min, coord_max, (255, 0, 0), 1)
        cv2.putText(image, f"Score: {detection_score:.2f}", coord_min, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        joint_visible = detection_keypoints_score > joint_threshold
        detection_keypoints = detection_keypoints.reshape(17, 2)

        for joint, joint_score in zip(detection_keypoints, detection_keypoints_score):
            if joint_score > joint_threshold:
                cv2.circle(image, scale_coord(joint), 4, (255, 0, 255), -1)

        for joint0, joint1 in JOINT_PAIRS:
            if joint_visible[joint0] and joint_visible[joint1]:
                cv2.line(image, scale_coord(detection_keypoints[joint0]),
                         scale_coord(detection_keypoints[joint1]), (255, 0, 255), 3)

def draw_predictions(request):
    global fall_detected, last_fall_time
    with MappedArray(request, 'main') as m:
        predictions = last_predictions
        if predictions:
            visualize_pose_estimation_result(predictions, m.array, model_size)
            if is_fallen(predictions, hip_y_buffer, model_size) and not fall_detected:
                cv2.putText(m.array, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                fall_detected = True
                last_fall_time = time.time()
                threading.Thread(target=handle_fall_alert, daemon=True).start()


last_predictions = None

try:
    # Kinesis 스트리밍 시작
    start_streaming()

    with Hailo(args.model) as hailo:
        main_size = (1024, 768)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)

        with Picamera2() as picam2:
            main = {'size': main_size, 'format': 'XRGB8888'}
            lores = {'size': lores_size, 'format': 'RGB888'}
            config = picam2.create_video_configuration(main, lores=lores)
            picam2.configure(config)

            picam2.start()
            picam2.pre_callback = draw_predictions

            while True:
                frame = picam2.capture_array('lores')
                raw_detections = hailo.run(frame)
                last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)

                visualize_pose_estimation_result(last_predictions, frame, model_size)
                if is_fallen(last_predictions, hip_y_buffer, model_size) and not fall_detected:
                    cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    fall_detected = True
                    last_fall_time = time.time()
                    threading.Thread(target=handle_fall_alert, daemon=True).start()

                if fall_detected and time.time() - last_fall_time < 20:
                    remaining_time = int(20 - (time.time() - last_fall_time))
                    cv2.putText(frame, f"Press button ({remaining_time}s)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('cam', frame)
                cv2.waitKey(1)
finally:
    stop_streaming()
    GPIO.cleanup()
