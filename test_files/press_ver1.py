import cv2
import mediapipe as mp
import numpy as np

# init mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands module setup
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# init webcam
cap = cv2.VideoCapture(0)

# init vars
pressed = [False] * 10
released = [False] * 10
prev_pos = [None] * 10
smoothed_pos = [None] * 10

down_threshold = 11 ** 3
mov_threshold = 2
frames_to_confirm = 3    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot retrieve frames from the camera.")
        break
    
    # preprocess frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # extract fingertip positions
            h, w, _ = frame.shape
            fingertip_indices = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]
            fingertips = []

            for idx, landmark_idx in enumerate(fingertip_indices):
                landmark = hand_landmarks.landmark[landmark_idx]
                fingertips.append((int(landmark.x * w), int(landmark.y * h)))
            
            for i, pos in enumerate(fingertips):
                # smooth fingertip positions -> reduce noise
                if smoothed_pos[i] is None:
                    smoothed_pos[i] = pos[1]
                else:
                    alpha = 0.5
                    smoothed_pos[i] = alpha * pos[1] + (1 - alpha) * smoothed_pos[i]

                # calculate velocity
                if prev_pos[i] is None:
                    prev_pos[i] = pos
                    continue

                velocity = (smoothed_pos[i] - prev_pos[i][1])**3

                # set velocity to zero if no significant movement
                if abs(velocity) < mov_threshold:
                    velocity = 0

                prev_pos[i] = pos

                # check for pressing actions
                if velocity > down_threshold:
                    pressed[i] = True

                elif velocity <= 0:
                    pressed[i] = False

                color = (0, 255, 0) if pressed[i] else (0, 0, 255)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, color, -1)
                cv2.putText(frame, f"{int(smoothed_pos[i])}", (pos[0], pos[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"{int(velocity)}", (pos[0], pos[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    
    cv2.imshow('Press Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# exit
hands.close()
cap.release()
cv2.destroyAllWindows()

# check for lowest y
