import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from datetime import datetime, timedelta
import pyttsx3
import threading
from queue import Queue

# Load models
model_hand = 'C:/Projects/FinalProject/excesise_proj/hand_ex_model/hand_exce2.pkl'
model_filename = 'C:/Projects/FinalProject/excesise_proj/leg_exces/leg_position_knn.joblib'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33 + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

with open(model_hand, 'rb') as file:
    model_hand = pickle.load(file)

with open(model_filename, 'rb') as file:
    leg_pos = pickle.load(file)

# Initialize variables
cap = cv2.VideoCapture(0)
count = 0
state = ''

class_to_action = {
    'nutral': 'nutral',
    'mid': 'mid',
    'wide': 'wide'
}

# Timer variables
start_time = datetime.now()
duration = timedelta(minutes=10)  # 10 minute timer

# Initialize text-to-speech engine and queue
engine = pyttsx3.init()
speech_queue = Queue()

# Function to handle text-to-speech
def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Flags to track if messages have been spoken
legs_mid_message_spoken = False

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
            state_class = model_hand.predict(x)[0]
            state_prob = model_hand.predict_proba(x)[0]

            leg_class = leg_pos.predict(x)[0]
            leg_prob = leg_pos.predict_proba(x)[0]

            if state_class == 'down' and state_prob[state_prob.argmax()] >= 0.7:
                state = 'down'
            elif state == 'down' and state_class == 'up' and state_prob[state_prob.argmax()] >= 0.7:
                state = 'up'
                count += 1

            if leg_class in class_to_action and leg_prob[leg_prob.argmax()] >= 0.7:
                state1 = class_to_action[leg_class]

            if leg_class == 'wide':
                cv2.putText(image, 'Keep your legs in mid range', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                if not legs_mid_message_spoken:
                    speech_queue.put('Keep your legs in mid range')
                    legs_mid_message_spoken = True
            else:
                legs_mid_message_spoken = False

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            cv2.putText(image, 'HAND STATE', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, state_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'COUNT', (180, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count), (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (400, 0), (560, 40), (0, 255, 0), -1)
            cv2.putText(image, 'LEG POS', (455, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, leg_class.split(' ')[0], (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        # Calculate remaining time
        elapsed_time = datetime.now() - start_time
        remaining_time = duration - elapsed_time
        minutes, seconds = divmod(remaining_time.seconds, 60)
        timer_text = f"{minutes:02}:{seconds:02}"

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Display the timer in the bottom left corner
        timer_position = (10, frame_height - 10)  # 10 pixels from the left and bottom edges
        cv2.putText(image, timer_text, timer_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if remaining_time.total_seconds() <= 0:
            break

cap.release()
cv2.destroyAllWindows()

# Stop the speech worker
speech_queue.put(None)
speech_thread.join()
