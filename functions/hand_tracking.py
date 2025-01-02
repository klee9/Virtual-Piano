import cv2
import mediapipe as mp

# initialize variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

landmark_color = (0, 0, 255)
connection_color = (0, 0, 0) 
landmark_style = mp_drawing.DrawingSpec(color=landmark_color, thickness=5, circle_radius=5)
connection_style = mp_drawing.DrawingSpec(color=connection_color, thickness=5)

# launch webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # hand pose estimation
    results = hands.process(rgb_f)

    # landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style 
            )
          
    cv2.imshow("Hand Pose Estimation", frame)

    # exit upon pressing esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

# terminate
cap.release()
cv2.destroyAllWindows()
