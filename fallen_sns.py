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
import gi
import os



gi.require_version('Gst','1.0')
from gi.repository import Gst, GstApp, GLib
Gst.init(None)
#aws에 kvssink로 스트리밍하기 위한 부분
gst_is = f"appsrc name=source is-live=true format=time ! videoconvert ! video/x-raw, format=I420 , width=640, height=640, framerate=30/1 ! \
x264enc byte-stream=true tune=zerolatency bitrate=500 speed-preset=ultrafast ! video/x-h264, level=4 ! h264parse ! video/x-h264, stream-format=avc, alignment=au ! kvssink stream-name=STREAM \
access-key={os.environ['AWS_ACCESS_KEY_ID']} secret-key={os.environ['AWS_SECRET_ACCESS_KEY']} aws-region={os.environ['AWS_DEFAULT_REGION']}"

testsink = f"videotestsrc is-live=true ! videoconvert ! video/x-raw, format=I420, width=640, height=480 ! x264enc tune=zerolatency bitrate=500 ! video/x-h264, stream-format=avc, alignment=au ! \
kvssink stream-name=STREAM access-key={os.environ['AWS_ACCESS_KEY_ID']} secret-key={os.environ['AWS_SECRET_ACCESS_KEY']} aws-region={os.environ['AWS_DEFAULT_REGION']}"
pipeline= Gst.parse_launch(gst_is)
appsrc=pipeline.get_by_name("source")

appsrc.set_property("format", Gst.Format.TIME)
appsrc.set_property("is-live",True)
appsrc.set_property("block",False) # 버퍼가 찬 경우 블록

caps = Gst.Caps.from_string("video/x-raw, format=RGB, width=640, height=640, framerate=30/1")
appsrc.set_caps(caps)

pipeline.set_state(Gst.State.PLAYING)
def push_data ( appsrc, endi, inp, _pts = 0): # appsrc - 싱크, endi - 종료 여부(false 시 종료), inp - 프레임 데이터
    pts = _pts
    duration = int(1e9/30) #기본적으로 30프레임 고정
    
    if endi:
        frame = inp
        frame_data = frame.tobytes()
        Buffer = Gst.Buffer.new_wrapped(frame_data)
        #Buffer.pts = pts # 타임스탬프
        #Buffer.duration = duration
        #if appsrc is None:
        #    print("NONE")
        #else:
        #    print("APPSRC")
        ret = appsrc.emit("push-buffer", Buffer)
        #print(ret)
        #if ret != Gst.FlowReturn.OK:
        #    print(f"Error Pushing Buffer: {ret}")

            
        pts +=duration
        #bt+=duration
        #frame_count +=1
        return True
    
    appsrc.emit("end-of-stream")
    return False
# -------------------------------------------------- #

parser = argparse.ArgumentParser(description='Pose estimation using Hailo')
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
sns_client = boto3.client('sns', region_name='us-east-1')  # 지역 수정
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:FallDetectionTopic'  # ARN 수정
BUTTON_PIN = 20  # 버튼 GPIO 핀
BUZZER_PIN = 21  # 부저 GPIO 핀

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
buzzer = GPIO.PWM(BUZZER_PIN, 1000)  # PWM 초기화 (주파수 1000Hz)

# 상태 변수
fall_detected = False
button_pressed = False
last_fall_time = 0

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
    buzzer.start(50)  # 부저 소리 시작
    print("Buzzer alert started")

def stop_alert_sound():
    buzzer.stop()  # 부저 소리 중지
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
    """스레드에서 부저와 버튼 처리"""
    global button_pressed, fall_detected
    play_alert_sound()
    start_time = time.time()

    while time.time() - start_time < 20:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # 버튼 눌림
            button_pressed = True
            stop_alert_sound()
            print("Button pressed, fall confirmed as safe")
            break
        time.sleep(0.1)

    if not button_pressed:
        stop_alert_sound()
        send_sns_notification()
    fall_detected = False  # 상태 초기화

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
        #cv2.putText(image, f"Score: {detection_score:.2f}", coord_min, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

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
                
                push_data(appsrc, True, frame) #스트리밍

                visualize_pose_estimation_result(last_predictions, frame, model_size)
                if is_fallen(last_predictions, hip_y_buffer, model_size) and not fall_detected:
                    cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    fall_detected = True
                    last_fall_time = time.time()
                    threading.Thread(target=handle_fall_alert, daemon=True).start()

                # 대기 시간 표시 (선택적)
                if fall_detected and time.time() - last_fall_time < 20:
                    remaining_time = int(20 - (time.time() - last_fall_time))
                    cv2.putText(frame, f"Press button ({remaining_time}s)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('cam', frame)
                cv2.waitKey(1)
except KeyboardInterrupt:
    appsrc.emit("end-of-stream")
    GPIO.cleanup()
