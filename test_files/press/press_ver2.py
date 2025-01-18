import cv2
import time
import mediapipe as mp
import numpy as np

# mediapipe stuff
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_cnt = 2

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=hands_cnt,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

fingertip_indices = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

# hands related stuff
y_base = 200
new_y = [float('inf')] * 5 * hands_cnt
pressed = [False] * 5 * hands_cnt
prev_pos = [None] * 5 * hands_cnt
smoothed_pos = [None] * 5 * hands_cnt
fingers_base = [None] * 5 * hands_cnt
base_set = False
color = None
alpha = 0.55

# start timer
start_time = time.time()

# webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open the webcam.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # preprocess frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # find hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        fingertips = []

        # initialize lowest points for each hand
        lowest_left_y = float('inf')
        lowest_right_y = float('inf')
        lowest_left_idx = None
        lowest_right_idx = None

        for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            label = handedness.classification[0].label  # "Left" or "Right"

            for j, landmark_idx in enumerate(fingertip_indices):
                idx = i * 5 + j  # global finger index

                landmark = hand_landmarks.landmark[landmark_idx]
                px, py = int(landmark.x * w), int(landmark.y * h)
                fingertips.append((px, py))
                
                # set base positions after 3 seconds
                if not base_set and time.time() - start_time >= 3:
                    if len(fingertips) == 10:
                        for k in range(len(fingertips)):
                            fingers_base[k] = smoothed_pos[k] if smoothed_pos[k] is not None else py
                        base_set = True
                        print("Base positions set.")

                # smooth the actual y coordinates
                if smoothed_pos[idx] is None:
                    smoothed_pos[idx] = py
                else:
                    smoothed_pos[idx] = int(alpha * smoothed_pos[idx] + (1 - alpha) * py)

                # display actual landmakrs (for debugging purposes)
                cv2.circle(frame, (px, smoothed_pos[idx]), 8, (255, 0, 0), -1)

                # calculate dy
                if prev_pos[idx] is not None:
                    dy = smoothed_pos[idx] - prev_pos[idx]
                else:
                    dy = 0
                if dy < 7: 
                    dy = 0

                prev_pos[idx] = smoothed_pos[idx]
                pressed[idx] = True if fingers_base[idx] is not None and abs(smoothed_pos[idx] - fingers_base[idx]) < 22 else False

                new_y[idx] = (int(y_base + 8 * dy))
                if (new_y[idx] > y_base + 1) and pressed[idx]:
                    color = (0,255,0)
                elif not base_set:
                    color = (255,0,0)
                else:
                    color = (0, 0, 255)
                cv2.circle(frame, (px, new_y[idx]), 8, color, -1)

        # find the lowest point
        for idx in range(5):
            if smoothed_pos[idx] is not None and smoothed_pos[idx] < lowest_left_y:
                lowest_left_y = smoothed_pos[idx]
                lowest_left_idx = idx

        for idx in range(5, 10):
            if smoothed_pos[idx] is not None and smoothed_pos[idx] < lowest_right_y:
                lowest_right_y = smoothed_pos[idx]
                lowest_right_idx = idx

    # display frame
    cv2.imshow('Press Test', frame)

    # exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
