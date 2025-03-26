from detectfall import FallDetector
from gpi import gpioController
from girun import grn
from snssender import snsSender

import argparse
import cv2
from pose_utils import postproc_yolov8_pose
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo
import time
import threading

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

JOINT_PAIRS = [[NOSE, L_EYE], [L_EYE, L_EAR], [NOSE, R_EYE], [R_EYE, R_EAR],
               [L_SHOULDER, R_SHOULDER],
               [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST],
               [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP],
               [L_HIP, L_KNEE], [R_HIP, R_KNEE], [L_KNEE, L_ANKLE], [R_KNEE, R_ANKLE]]

GIS = grn("gst_is5")
GPI = gpioController()
FD = FallDetector()
SNS = snsSender()

parser = argparse.ArgumentParser(description='Pose estimation using Hailo')
parser.add_argument('-m', '--model', help="HEF file path", default="/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef")
args = parser.parse_args()

last_predictions = None

# 상태 변수
fall_detected = False
button_pressed = False
last_fall_time = 0

def handle_fall_alert():
    global button_pressed, fall_detected
    GPI.make_buzz()
    start_time = time.time()

    while time.time()-start_time<20:
        if(GPI.get_button()):
            button_pressed = True
            GPI.stop_buzz()
            break
        time.sleep(0.1)

    if not button_pressed:
        GPI.stop_buzz()
        SNS.send_sns_message('fall detected')
    
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

try:
    with Hailo(args.model) as hailo:
        main_size = (1024,768)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)

        with Picamera2() as picam2:
            main = {'size': main_size, 'format': 'XRGB8888'}
            lores = {'size': lores_size, 'format': 'RGB888'}
            config = picam2.create_video_configuration(main, lores=lores)
            picam2.configure(config)

            picam2.start()
            #picam2.pre_callback = draw_predictions

            while True:
                frame = picam2.capture_array('lores')
                raw_detections = hailo.run(frame)
                last_predictions = postproc_yolov8_pose(1,raw_detections,model_size)

                GIS.push_data(frame)
                kps = FD.process_pose_data(last_predictions, model_size)
                visualize_pose_estimation_result(last_predictions,frame, model_size)

                if kps is not None:
                    LS, RS, LH, RH = kps
                    if (FD.detect_fall(LS,RS,LH,RH)) and not fall_detected:
                        
                        cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        fall_detected = True
                        last_fall_time = time.time()

                        threading.Thread(target=handle_fall_alert, daemon=True).start()
                
                cv2.imshow(frame)




except KeyboardInterrupt:
    GIS.end_stream()
    GPI.end_proc()
