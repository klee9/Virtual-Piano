import cv2
import time
import pygame
import mediapipe as mp
import numpy as np

def is_valid_quad(corners, min_area=1000, min_angle=45.0):
    if len(corners) != 4:
            return False
            
     # 면적 검사
    area = cv2.contourArea(corners)
    if area < min_area:
        return False
            
    # 각도 검사
    corners = corners.reshape(-1, 2)
    angles = []
    for i in range(4):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 4]
        pt3 = corners[(i + 2) % 4]
            
        # 두 벡터 계산
        v1 = pt1 - pt2
        v2 = pt3 - pt2
            
        # 각도 계산
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        angles.append(angle)
            
    # 모든 각도가 min_angle보다 커야 함
    return all(angle >= min_angle and angle <= 180-min_angle for angle in angles)

def preprocess_image(frame):
    # grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
    # find edges
    edges = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
        
    return dilated

def sort_points(pts):
    # y좌표로 정렬
    sorted_by_y = pts[np.argsort(pts[:, 1])]
        
    # 상단/하단 두 점씩 분리
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
        
    # x좌표로 정렬
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]
        
    return np.array([top_left, bottom_left, bottom_right, top_right])

# set up pygame
pygame.init()
pygame.display.set_caption('Virtual Piano')

screen_w, screen_h = 3456, 2234
screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
piano_layer = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
black_layer = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
black_layer.fill((0, 0, 0, 200))

piano = cv2.imread("/Users/klee9/Desktop/daiv/kirby/piano1.png")
piano_h, piano_w, _ = piano.shape

fixed_corners = None
warped_piano = None
running = True

# webcam & mediapipe
cap = cv2.VideoCapture(0)
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

# hand tracking stuff
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

if not cap.isOpened():
    print("Error: webcam failed to launch.")
    exit()

# launch window
while running:
    ret, frame = cap.read() 
    if not ret:
        print("Error: failed to grab frame.")
        break

    # preprocess frame
    h, w, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_layer = np.zeros((h, w, 4), dtype=np.uint8)
    temp_layer = np.zeros((h, w, 4), dtype=np.uint8)

    results = hands.process(frame)

    # handle key presses
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # quit on pressing 'q'
                running = False
            elif event.key == pygame.K_p:  # set fixed corners on 'p' press
                if fixed_corners is None and is_valid_quad(corners):
                    fixed_corners = corners
                    print("Fixed corners set.")

    # detect landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                # get the start and end points of the connection
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]

                # convert normalized coordinates to pixel coordinates
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # add neon effect
                glow_color = (255, 0, 255, 255)
                for thickness in [22, 17, 12]:
                    cv2.line(hand_layer, start_point, end_point, glow_color, thickness, cv2.LINE_AA)
                    cv2.circle(hand_layer, start_point, thickness, glow_color, cv2.LINE_AA)

                cv2.line(temp_layer, start_point, end_point, (255, 255, 255, 255), 12, cv2.LINE_AA)
                cv2.circle(temp_layer, start_point, 12, (255, 255, 255, 255), -1)
                if start_point != end_point:
                    cv2.circle(temp_layer, end_point, 12, (255, 255, 255, 255), -1)

    # neon effect
    hand_layer = cv2.GaussianBlur(hand_layer, (51, 51), 30)

    # find contours
    if fixed_corners is None:
        processed_frame = preprocess_image(frame)
        ret, thresh = cv2.threshold(processed_frame, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find the largest space enclosed by the contours
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

        # ensure it's a rectangle
        if len(approx_corners) == 4:
            corners = approx_corners.reshape(4, 2)
            corners = sort_points(corners)

            if is_valid_quad(corners):
                cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0, 255), 5)

    else:
        piano_pts = np.array([[0, 0], [0, piano_h], [piano_w, piano_h], [piano_w, 0]], dtype=np.float32)
        doc_pts = np.float32(corners)

        # warp perspective
        matrix = cv2.getPerspectiveTransform(piano_pts, doc_pts)
        warped_piano = cv2.warpPerspective(piano, matrix, (frame.shape[1], frame.shape[0]))
        mask = cv2.cvtColor(warped_piano, cv2.COLOR_BGR2GRAY) > 0
        result = frame.copy()
        result[mask] = cv2.addWeighted(frame, 0.0, warped_piano, 1.0, 0)[mask]
        piano_layer = pygame.surfarray.make_surface(result)
        piano_layer = pygame.transform.rotate(piano_layer, -90)

    # create a pygame surface
    surface = pygame.surfarray.make_surface(frame)
    surface = pygame.transform.rotate(surface, -90)
    hand_layer = pygame.image.frombuffer(hand_layer.tobytes(), (w, h), "RGBA")
    temp_layer = pygame.image.frombuffer(temp_layer.tobytes(), (w, h), "RGBA")
    hand_layer = pygame.transform.flip(hand_layer, 1, 0)
    temp_layer = pygame.transform.flip(temp_layer, 1, 0)

    # stack surfaces
    screen.blit(surface, (0, 0))
    screen.blit(piano_layer, (0, 0))
    screen.blit(hand_layer, (0, 0))
    screen.blit(temp_layer, (0, 0))
    pygame.display.flip()

# release resources
cap.release()
pygame.quit()
