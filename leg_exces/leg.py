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
import joblib

if _name_ == "_main_":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Save the model
    model_path = 'leg_position.pkl'

    # Load the model
    model = joblib.load(model_path)

    cap = cv2.VideoCapture(0)
    state = ''

    # Mapping of classes to actions
    class_to_action = {
        'nutral': 'nutral',
        'mid': 'mid',
        'wide': 'wide'
    }
    landmarks = ['class']

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
                state_class = model.predict(x)[0]
                state_prob = model.predict_proba(x)[0]

                if state_class in class_to_action and state_prob[state_prob.argmax()] >= 0.7:
                    state = class_to_action[state_class]

                cv2.rectangle(image, (0,0), (250,60), (245,117,16), -1)

                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, state_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(state_prob[np.argmax(state_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            except Exception as e:
                print(f"Error: {e}")

            cv2.imshow('webcam feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()