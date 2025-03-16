import cv2
import mediapipe as mp
import numpy as np
import csv

import os
import numpy as np
import pandas as pd 
import pickle
import mediapipe as mp


model_filename = 'C:/Projects/FinalProject/excesise_proj/head_exces/head_exces1.pkl'
model_hip_on = 'C:/Projects/FinalProject/excesise_proj/hand_on_hip/hand_on_hip.pkl'

with open(model_filename, 'rb') as file:
    head_model = pickle.load(file)

with open(model_hip_on, 'rb') as file:
    model_hip_on = pickle.load(file)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# Initialize variables
cap = cv2.VideoCapture(0)
state = ''
state2 = ''

# Mapping of classes to actions
class_to_action = {
    'forward': 'forward',
    'left': 'left',
    'right': 'right',
    'up': 'up',
    'down': 'down'
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
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            x = pd.DataFrame([row], columns=landmarks[1:])
            state_class = head_model.predict(x)[0]
            state_prob = head_model.predict_proba(x)[0]
            
            hip_on_class = model_hip_on.predict(x)[0]
            hip_on_prob = model_hip_on.predict_proba(x)[0]

            if state_class in class_to_action and state_prob[state_prob.argmax()] >= 0.7:
                state = class_to_action[state_class]

            if hip_on_class == 'on' and hip_on_prob[hip_on_prob.argmax()] >= 0.7:
                state2 = 'on'
            elif state2 == 'on' and hip_on_class == 'off' and hip_on_prob[hip_on_prob.argmax()] >= 0.7:
                state2 = 'off'
            if hip_on_class == 'off' and hip_on_prob[hip_on_prob.argmax()] >= 0.7:
                cv2.putText(image, 'Keep your hand on hip', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



            cv2.rectangle(image, (0, 0), (200, 40), (245, 117, 16), -1)  # First rectangle

            cv2.putText(image, 'HEAD POSITION', (55, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, state_class.split(' ')[0], (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            
            cv2.rectangle(image, (220, 0), (380, 40), (16, 117, 245), -1)  # Second rectangle

            cv2.putText(image, 'HAND ON HIP', (275, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, hip_on_class.split(' ')[0], (270, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            
        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()