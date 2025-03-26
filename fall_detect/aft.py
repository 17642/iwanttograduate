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
                if kps is not None:
                    LS, RS, LH, RH = kps
                    if (FD.detect_fall(LS,RS,LH,RH)) and not fall_detected:
                        
                        cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        fall_detected = True
                        last_fall_time = time.time()

                        threading.Thread(target=handle_fall_alert, daemon=True).start()




except KeyboardInterrupt:
    GIS.end_stream()
    GPI.end_proc()