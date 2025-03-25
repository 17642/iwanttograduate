from math import sqrt
from collections import deque

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

JOINT_PAIRS = [[NOSE, L_EYE], [L_EYE, L_EAR], [NOSE, R_EYE], [R_EYE, R_EAR],
               [L_SHOULDER, R_SHOULDER],
               [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST],
               [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP],
               [L_HIP, L_KNEE], [R_HIP, R_KNEE], [L_KNEE, L_ANKLE], [R_KNEE, R_ANKLE]]

class FallDetector:
    def __init__(self):
        # 사전 임계값 수치
        self.HIP_DROP_THRESHOLD = 50
        self.SHOULDER_TWIST_THRESHOLD = 30
        self.HEIGHT_DIFF_THRESHOLD = 100
        self.BUFFER_SIZE = 15

        #상태 변수
        self.fall_detected = False
        self.button_pressed = False
        self.last_fall_time = False

        #버퍼
        self.hip_y_buffer = deque(15)
        
    def process_pose_data(self,results, model_size, detection_threshold = 0.5, joint_threshold = 0.5):

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

            if not (joint_visible[L_SHOULDER] and joint_visible[R_SHOULDER] and
                joint_visible[L_HIP] and joint_visible[R_HIP]):
                return None
            
            return left_shoulder, right_shoulder, left_hip, right_hip
        return None
    
    def detect_fall(self, LS, RS, LH, RH):
        curr_hip_y = (LH[1] + RH[1]) / 2
        self.hip_y_buffer.append(curr_hip_y)

        hip_drop = False
        if len(self.hip_y_buffer) == self.hip_y_buffer.maxlen:
            oldest_hip_y = self.hip_y_buffer[0]
            hip_drop_speed = curr_hip_y - oldest_hip_y
            if hip_drop_speed > 50:
                hip_drop = True

        if hip_drop:
            shoulder_diff = abs(LS[1] - RS[1])
            shoulder_twist = shoulder_diff > 30
            shoulder_y = (LS[1] + RS[1]) / 2
            hip_y = (LH[1] + RH[1]) / 2
            height_diff = abs(shoulder_y - hip_y)
            similar_height = height_diff < 100

            if shoulder_twist or similar_height:
                return True
        
        return False