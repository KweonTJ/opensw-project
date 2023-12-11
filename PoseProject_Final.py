import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import time
import argparse

YOUR_TIME_THRESHOLD = 30
YOUR_THRESHOLD_ANGLE = 178

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

parser = argparse.ArgumentParser(description='Pose Project')
parser.add_argument('--camera_index', type=int, default=0, help='Index of the camera')
args = parser.parse_args()

def new_func(f):
    model = pickle.load(f)
    return model

with open('body_language.pkl', 'rb') as f:
    model = new_func(f)

def calculate_angle(a,b):
    a = np.array(a)
    b = np.array(b)
    
    radians = np.arctan2(b[1]-a[1], b[0]-a[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle 

def run():
    #VideoCapture의 인덱스를 명령행 인수로 받아오도록 변경
    cap = cv2.VideoCapture(args.camera_index)
    # Curl counter variables
    warning = False
    count = 0
    good_count = 0
    stretch_count = 0
    stand_count = 0
    start = time.time()     # 시작 시간 저장

    #Initialize body_language_class
    body_language_class = 'Initial' # 아무 상태로 초기화
    elapsed_minutes = 0
    shoulder_angle = 0

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            resize_frame = cv2.resize(frame ,None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) 
            
            # Recolor Feed
            image = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            # Display elapsed time on the screen
            elapsed_time = time.time() - start
            elapsed_minutes = elapsed_time // 60  # elapsed_time을 분 단위로

            cv2.putText(image, f'Time Elapsed: {int(elapsed_minutes)} minutes', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

             # 추가된 부분: 일정 시간 이상 앉아 있으면 일어나라는 문구를 표시
            if elapsed_minutes >= YOUR_TIME_THRESHOLD:
                cv2.putText(image, 'Please Stand UP',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                
             # 추가된 부분: 어깨 각도가 일정 수준 이하로 내려갔을 때 경고 표시
            if shoulder_angle < YOUR_THRESHOLD_ANGLE:
                warning = True
                cv2.putText(image, 'Warning: Your shoulder angle is too low!', 
                            (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
               
                # Calculate angle
                shoulder_angle = calculate_angle(left_shoulder, right_shoulder)

                # 디버깅 문구 추가
                print(f"Shoulder Angle: {shoulder_angle}, Body Language Class: {body_language_class}")


                # Display angle on the screen with color based on the condition
                if shoulder_angle <= 178:
                    angle_color = (0,0,255) #red color
                else:
                    angle_color = (0,0,0) # dark color

                #Display shoulder angle on the screen
                cv2.putText(image,f'Shoulder Angle: {round(shoulder_angle,2)}degrees', (10,100), cv2.FONT_HERSHEY_SIMPLEX,0.7,
                            angle_color, 2,cv2.LINE_AA)
                
                #Display elapsed time on the screen
                elapsed_time = time.time() - start
                elapsed_minutes = elapsed_time // 60 # elapsed_time을 분 단위로

                now = time.gmtime(elapsed_time)
                hour = now.tm_hour
                minutes = now.tm_min

                cv2.putText(image, f'Time Elapsed: {hour} hours {minutes} minutes', (10,25),
                           cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0), 1, cv2.LINE_AA)
               
                # 추가된 부분: 어깨 각도가 일정 수준 이하로 내려갔을 때 경고 표시
                if shoulder_angle < YOUR_THRESHOLD_ANGLE:
                    warning = True
                    cv2.putText(image, 'Warning: Your shoulder angle is too low!', 
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # Curl counter logic
                if shoulder_angle < 178 or body_language_class.split(' ')[0] == 'Bad':
                    count = count + 1
                    good_count = 0   

                # 디버깅 문구 추가
                    print(f"Shoulder Angle: {shoulder_angle}, Body Language Class: {body_language_class}")

                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concate rows
                row = pose_row+face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # Get status box
                cv2.rectangle(image, (0,0), (1000, 80), (128,128,128), -1)
                
                #Time
                now = time.gmtime(time.time())
            
                cv2.putText(image, 'Time', 
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                hour = now.tm_hour - start.tm_hour
                minutes = abs(now.tm_min - start.tm_min)
                
                cv2.putText(image, str(hour) +' : '+ str(minutes), 
                            (250,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                if minutes >= 5:
                    if body_language_class.split(' ')[0] == 'Stand' and round(body_language_prob[np.argmax(body_language_prob)],2) > 0.5:
                        stand_count += 1
                    if stand_count < 30:
                        cv2.putText(image, 'Please Stand UP', 
                            (350,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, 'Great!!', 
                                (350,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                        stand_count = 0
                        count = 0
                        stretch_count = 0
                        good_count = 0
                        start = time.gmtime(time.time())  
                        time.sleep(0.1)
                 
                #Warning
                if minutes < 5 and count > 10:
                    if body_language_class.split(' ')[0] == 'Stretch' and round(body_language_prob[np.argmax(body_language_prob)],2) > 0.2:
                        stretch_count += 1
                        
                    if good_count > 5 or stretch_count > 20:
                        cv2.putText(image, 'Great!!', 
                                (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                        count = 0
                        stretch_count = 0
                        good_count = 0
                        time.sleep(0.5)
                        
                    else:
                        cv2.putText(image, 'Please Straighten UP', 
                                    (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)  
                        
                # Display Class
                cv2.putText(image, 'Status'
                            , (150,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (150,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (280,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, str(round(shoulder_angle,2))
                            , (850,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                pass
                            
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run()