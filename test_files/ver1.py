import cv2
import pygame
import mediapipe as mp
import numpy as np


def sort_points(pts):
    '''
    sort in the order of: top-left, bot-left, bot-right, top-right
    '''
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
black_layer = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
black_layer.fill((0, 0, 0, 200))

piano = pygame.image.load("/Users/klee9/Desktop/daiv/kirby/imgs/piano_ui.jpg")
fixed_corners = None

running = True

# webcam & mediapipe
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_layer = np.zeros((h, w, 4), dtype=np.uint8)
    temp_layer = np.zeros((h, w, 4), dtype=np.uint8)

    # mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    scaled_mouse_x = int(mouse_x * w / screen_w)
    scaled_mouse_y = int(mouse_y * h / screen_h)

    results = hands.process(frame)

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
        frame_edges = cv2.Canny(frame, 100, 200)
        ret, thresh = cv2.threshold(frame_edges, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find the largest space enclosed by the contours
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        cv2.drawContours(temp_layer, [largest_contour], -1, (255, 255, 255, 255), 8)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

        # ensure it's a rectangle
        if len(approx_corners) == 4:
            corners = approx_corners.reshape(4, 2)
            corners = sort_points(corners)
            for corner in corners:
                cv2.circle(temp_layer, tuple(corner), 5, (255, 0, 0, 255), -1)

            # resize piano
            img_w = max(piano.get_width(), abs(corners[3,0] - corners[0,0]))
            img_h = abs(corners[0][1] - corners[1][1])
            img_scale = (img_w, img_h)
            piano = pygame.transform.scale(piano, img_scale)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    if fixed_corners is None and len(approx_corners) == 4:
                        fixed_corners = corners
                        
                        print(f"Fixed corners set: {fixed_corners}")

            # Key release events (optional)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_p:
                    print("P key released")

    # create a pygame surface
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    hand_layer = pygame.image.frombuffer(hand_layer.tobytes(), (w, h), "RGBA")
    temp_layer = pygame.image.frombuffer(temp_layer.tobytes(), (w, h), "RGBA")

    # stack surfaces
    screen.blit(surface, (0, 0))
    screen.blit(black_layer, (0, 0))

    # for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_RETURN:
    #                 screen.blit(piano, (int((corners[0,0]+corners[3,0])/2)-img_w, corners[0, 1]))

    screen.blit(hand_layer, (0, 0))
    screen.blit(temp_layer, (0, 0))
    pygame.display.flip()

    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# release resources
cap.release()
pygame.quit()

'''
1. find planar object
2. draw a white outline around it (add glowing effect)
3. display piano UI
'''
