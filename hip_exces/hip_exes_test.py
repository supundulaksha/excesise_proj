import cv2
import mediapipe as mp
import numpy as np
import csv

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

import pickle
model_file = 'C:/Projects/FinalProject/excesise_proj/hip_exces/hip_exces_latest.pkl'


with open(model_file, 'rb') as file:
    model_hip = pickle.load(file)



# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
cap = cv2.VideoCapture(0)
count = 0

state1 =''
state2=''
state3 = 'midle'

class_to_action = {
    'nutral': 'nutral',
    'mid': 'mid',
    'wide': 'wide'
}

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            x = pd.DataFrame([row], columns=landmarks[1:])
            

            state_class = model_hip.predict(x)[0]
            state_prob = model_hip.predict_proba(x)[0]

           

            if state_class == 'left' and state_prob[state_prob.argmax()] >= 0.7:
                state3 = 'left'
            elif state3 == 'left' and state_class == 'midle' and state_prob[state_prob.argmax()] >= 0.7:
                state3 = 'midle'
                count += 1
            elif state3 == 'midle' and state_class == 'right' and state_prob[state_prob.argmax()] >= 0.7:
                state3 = 'right'
            


            
            cv2.rectangle(image, (0, 0), (200, 40), (245, 117, 16), -1)  # First rectangle

            cv2.putText(image, 'HIP CLASS', (55, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, state_class.split(' ')[0], (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

           
         
        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
