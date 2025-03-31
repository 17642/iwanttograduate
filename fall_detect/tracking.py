import RPi.GPIO as GPIO
import time

class Tracker:
    def __init__(self, pan_pin=18, tilt_pin=19):
        # 서보모터 핀 설정
        self.SERVO_PAN_PIN = pan_pin
        self.SERVO_TILT_PIN = tilt_pin
        
        # GPIO 초기화
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.SERVO_PAN_PIN, GPIO.OUT)
        GPIO.setup(self.SERVO_TILT_PIN, GPIO.OUT)
        
        # PWM 설정 (50Hz)
        self.pan_servo = GPIO.PWM(self.SERVO_PAN_PIN, 50)
        self.tilt_servo = GPIO.PWM(self.SERVO_TILT_PIN, 50)
        self.pan_servo.start(7.5)  # 중립 (90도)
        self.tilt_servo.start(7.5)
        
        # PID 변수
        self.pan_error_prev = 0
        self.tilt_error_prev = 0
        self.pan_integral = 0
        self.tilt_integral = 0
        self.Kp = 0.02  # 비례 gain
        self.Ki = 0.005  # 적분 gain
        self.Kd = 0.01  # 미분 gain
        
        # 추적 대상 ID
        self.tracked_person_id = None

    def track_person(self, results, image, model_size, detection_threshold=0.5):
        """사람 추적 및 서보모터 제어"""
        image_size = (image.shape[1], image.shape[0])
        
        def scale_coord(coord):
            return tuple([int(c * t / f) for c, f, t in zip(coord, model_size, image_size)])

        bboxes, scores = results['bboxes'], results['scores']
        box, score = bboxes[0], scores[0]
        
        tracked_center = None
        for idx, (detection_box, detection_score) in enumerate(zip(box, score)):
            if detection_score < detection_threshold:
                continue
                
            coord_min = scale_coord(detection_box[:2])
            coord_max = scale_coord(detection_box[2:])
            center_x = (coord_min[0] + coord_max[0]) // 2
            center_y = (coord_min[1] + coord_max[1]) // 2

            # 첫 감지 시 추적 대상 선정
            if self.tracked_person_id is None and detection_score >= detection_threshold:
                self.tracked_person_id = idx
            if idx == self.tracked_person_id:
                tracked_center = (center_x, center_y)
                cv2.rectangle(image, coord_min, coord_max, (0, 255, 0), 2)  # 추적 대상 초록색
            else:
                cv2.rectangle(image, coord_min, coord_max, (255, 0, 0), 1)  # 기타 빨간색

        # 추적 대상이 없으면 초기화
        if tracked_center is None and not any(score > detection_threshold):
            self.tracked_person_id = None

        # 서보모터 제어
        if tracked_center:
            self.control_servos(tracked_center, image_size[0], image_size[1])
        
        return tracked_center

    def control_servos(self, center, image_width, image_height):
        """PID 제어로 상하좌우 서보모터 조정"""
        center_x, center_y = center
        mid_x, mid_y = image_width // 2, image_height // 2
        
        # 오류 계산
        pan_error = center_x - mid_x
        tilt_error = mid_y - center_y  # Y축은 위가 작고 아래가 큼
        
        # PID 계산
        self.pan_integral += pan_error
        self.tilt_integral += tilt_error
        pan_derivative = pan_error - self.pan_error_prev
        tilt_derivative = tilt_error - self.tilt_error_prev
        
        pan_output = (self.Kp * pan_error) + (self.Ki * self.pan_integral) + (self.Kd * pan_derivative)
        tilt_output = (self.Kp * tilt_error) + (self.Ki * self.tilt_integral) + (self.Kd * tilt_derivative)
        
        # 현재 듀티 사이클 (중립 7.5 가정)
        pan_duty = 7.5 + pan_output
        tilt_duty = 7.5 + tilt_output
        
        # 듀티 사이클 제한 (2.5~12.5, 약 -90도~90도)
        pan_duty = max(2.5, min(12.5, pan_duty))
        tilt_duty = max(2.5, min(12.5, tilt_duty))
        
        # 서보모터 제어
        self.pan_servo.ChangeDutyCycle(pan_duty)
        self.tilt_servo.ChangeDutyCycle(tilt_duty)
        time.sleep(0.05)  # 서보 이동 대기
        self.pan_servo.ChangeDutyCycle(0)
        self.tilt_servo.ChangeDutyCycle(0)
        
        # 이전 오류 저장
        self.pan_error_prev = pan_error
        self.tilt_error_prev = tilt_error

    def cleanup(self):
        """GPIO 정리"""
        self.pan_servo.stop()
        self.tilt_servo.stop()