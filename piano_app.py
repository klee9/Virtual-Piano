import cv2
import numpy as np
import mediapipe as mp
from pygame import mixer
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def find_document_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None


def order_points(pts):
    try:
        # convert points to float32 if they aren't already
        pts = np.float32(pts)
        
        # initialize rectangle coordinates
        rect = np.zeros((4, 2), dtype=np.float32)
    
        # calculate sum of x and y coordinates
        s = np.sum(pts, axis=1)
        rect[3] = pts[np.argmin(s)]  # bottom-left
        rect[1] = pts[np.argmax(s)]  # bottom-right
        
        # calculate difference between x and y coordinates
        diff = np.diff(pts, axis=1)
        rect[2] = pts[np.argmin(diff)]  # bottom-right
        rect[0] = pts[np.argmax(diff)]  # top-left
        
        return rect
    except Exception as e:
        print(f"Error in order_points: {e}")
        return None
       

class PianoApp:
    def __init__(self):
        self.ar_piano = ARPiano()
        self.root = tk.Tk()
        self.root.title("AR Piano")
        self.w = 1280
        self.h = 720
        self.root.geometry(f"{self.w}x{self.h}")

        self.video_running = False
        self.cap = None
        self.frame = None
        self.fixed_corners = None
        self.calibration_started = False
        self.dy_incr = 0
        self.ret_incr = 0
        self.thresholds = [None] * 2
        self.corners = None
        self.ordered_corners = None

        self.bg_image = Image.open("bg.jpg")
        self.bg_image = self.bg_image.resize((self.w, self.h+20))
        self.bg_image = ImageTk.PhotoImage(self.bg_image)

        self.root.bind("<space>", self.space_key)
        self.root.bind("<w>", self.ret_up)
        self.root.bind("<s>", self.ret_down)
        self.root.bind("<a>", self.dy_up)
        self.root.bind("<d>", self.dy_down)
        self.root.bind("<r>", self.reset)
        self.root.bind("<Escape>", self.exit_app)

        self.start_screen()

    def start_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        # background image
        canvas = tk.Canvas(self.root, width=self.w, height=self.h)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

        canvas.tag_bind(canvas.create_text(self.w - 530, self.h - 500, text="Play", fill="black", font=("Helvetica", 42), anchor="w"), "<Button-1>", self.launch_main_app)
        canvas.tag_bind(canvas.create_text(self.w - 530, self.h - 500 + 42 + 50, text="How To Play", fill="black", font=("Helvetica", 42), anchor="w"), "<Button-1>", self.how_to_play)
        canvas.tag_bind(canvas.create_text(self.w - 530, self.h - 500 + 84 + 100, text="Exit", fill="black", font=("Helvetica", 42), anchor="w"), "<Button-1>", self.stop_video)

        canvas.pack(pady=10)

    def how_to_play(self, event):
        for widget in self.root.winfo_children():
            widget.destroy()

        Label(self.root, text="How to Play", font=("Helvetica", 20)).pack(pady=20)
        Label(self.root, text="1. 종이 등 평평한 물체를 준비해 주세요.\n"
                              "2. 물체의 모서리에 초록색 점이 표시되면 스페이스바를 눌러서 위치를 고정해 주세요.\n"
                              "3. 5초 동안 손의 위치를 측정합니다. \"Calibrating\" 문구가 사라지기 전까지 손을 움직이지 말아 주세요.\n"
                              "4. 이제 피아노를 칠 수 있습니다~! \n\n"
                              "W: 기준값 인식 범위 증가\n"
                              "S: 기준값 인식 범위 감소\n"
                              "A: y 변화량 임계값 증가\n"
                              "D: y 변화량 임계값 감소"
                              , 
                              font=("Helvetica", 14),
                              anchor="w",
                              justify="left").pack(pady=10)
        Button(self.root, text="Back", font=("Helvetica", 16), command=self.start_screen).pack(pady=20)

    def launch_main_app(self, event):
        for widget in self.root.winfo_children():
            widget.destroy()

        # webcam feed
        self.video_canvas = tk.Canvas(self.root, width=1280, height=720)
        self.video_canvas.pack()
        self.video_running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # use a separate thread for video processing
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()

    def video_loop(self):
        try:
            while self.video_running:
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Could not retrieve frame.")
                    break

                self.frame = cv2.flip(self.frame, 1)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                frame2 = self.frame.copy()

                # if positions are not fixed, find corners
                if self.fixed_corners is None:
                    self.corners = find_document_corners(self.frame)
                    if self.corners is not None:
                        ordered_corners = order_points(self.corners)
                        if ordered_corners is not None:
                            for corner in ordered_corners.astype(int):
                                cv2.circle(self.frame, tuple(corner), 8, (0, 255, 0), -1)
                
                if not self.calibration_started:
                    cv2.putText(self.frame, "Press SPACE to set piano position", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2)
                
                # if positions are fixed, disply the piano image
                if self.fixed_corners is not None:
                    self.frame = self.ar_piano.transform_keyboard(self.frame, self.fixed_corners)

                self.frame, hands_data, self.thresholds = self.ar_piano.hand_detector.find_hands(self.frame, frame2, self.ret_incr, self.dy_incr)

                if self.ar_piano.hand_detector.calibration_complete:
                    for hand_info in hands_data:
                        for x, y, is_pressed in hand_info:
                            self.ar_piano.check_key_press(x, y, is_pressed)

                img = Image.fromarray(self.frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.video_canvas.image = imgtk

                self.root.update_idletasks()
                self.root.update()

        except Exception as e:
            print(f"Error/Warning: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            mixer.quit()

    def stop_video(self, event):
        self.video_running = False
        self.start_screen()

    def run(self):
        self.root.mainloop()

    def space_key(self, event):
        if self.fixed_corners is None and not self.calibration_started:
            print("Positions fixed.")
            self.ordered_corners = order_points(self.corners)
            if self.ordered_corners is not None:
                self.fixed_corners = self.ordered_corners
                self.calibration_started = True
                self.ar_piano.hand_detector.start_calibration()

    def ret_up(self, event):
        self.ret_incr += 1
        if self.thresholds[0] is not None:
            cv2.putText(self.frame, f"Detection Range: {self.thresholds[0]+1}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (255, 255, 0), 2)

    def ret_down(self, event):
        if self.ret_incr > 0:
            self.ret_incr -= 1
            if self.thresholds[0] is not None:
                cv2.putText(self.frame, f"Detection Range: {self.thresholds[0]-1}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (255, 255, 0), 2)

    def dy_up(self, event):
        self.dy_incr += 1
        cv2.putText(self.frame, f"Movement Threshold: {self.thresholds[1]+1}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (255, 255, 0), 2)

    def dy_down(self, event):
        if self.dy_incr > 0:
            self.dy_incr -= 1
            cv2.putText(self.frame, f"Movement Threshold: {self.thresholds[1]-1}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (255, 255, 0), 2)

    def reset(self, event):
        self.fixed_corners = None
        self.calibration_started = False
        print("Position Reset - Press SPACE to set new position")

    def exit_app(self, event):
        print("Exiting...")
        self.root.quit()


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.fingertips = [4, 8, 12, 16, 20]
        self.position_history = {}
        self.base_positions = {}
        self.smoothed_positions = {}
        self.calibration_start = None
        self.calibration_complete = False
        self.alpha = 0.6

    def start_calibration(self):
        self.calibration_start = time.time()
        self.calibration_complete = False
        self.base_positions.clear()
        print("Calibrating...")

    def find_hands(self, frame, frame2, ret_incr, dy_incr):
        frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        h, w, _ = frame.shape
        hands_data = []
        thresholds = [None] * 2

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_info = []
                
                # draw landmarks and connections
                self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                         self.mp_hands.HAND_CONNECTIONS,
                                         self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3), # landmark color
                                         self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3))  # line color
                
                for tip_id in self.fingertips:
                    x = int(hand_landmarks.landmark[tip_id].x * w)
                    y = int(hand_landmarks.landmark[tip_id].y * h)
                    finger_id = f"{idx}_{tip_id}"
                    
                    if finger_id not in self.smoothed_positions:
                        self.smoothed_positions[finger_id] = y
                        
                    else:
                        prev_y = self.smoothed_positions[finger_id]
                        self.smoothed_positions[finger_id] = int(
                            self.alpha * prev_y + 
                            (1 - self.alpha) * y
                        )
                    y = self.smoothed_positions[finger_id]
                    
                    if self.calibration_start and not self.calibration_complete:
                        time_elapsed = time.time() - self.calibration_start
                        if time_elapsed <= 5:
                            remaining = max(0, 5 - int(time_elapsed))
                            cv2.putText(frame, f"Calibrating: {remaining}s", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (255, 255, 0), 2)
                        else:
                            if finger_id not in self.base_positions:
                                self.base_positions[finger_id] = y
                                print(f"Calibration - Finger ID: {finger_id}, Base Y: {y}")
                                
                                if len(self.base_positions) >= 10:
                                    if not self.calibration_complete:
                                        self.calibration_complete = True
                                        print("캘리브레이션 완료!")
                                        cv2.putText(frame, "Calibration Complete!", 
                                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  1, (0, 255, 0), 2)

                    is_pressed = False
                    ret_thres = 10 + ret_incr
                    dy_thres = 5 + dy_incr
                    thresholds[0], thresholds[1] = ret_thres, dy_thres
                    
                    if self.calibration_complete and finger_id in self.base_positions:
                        returned = abs(y - self.base_positions[finger_id]) < ret_thres
                        dy = y - prev_y
                        
                        if returned and dy > dy_thres:
                            is_pressed = True
                        
                    hand_info.append((x, y, is_pressed))
                
                hands_data.append(hand_info)
                
                # make the pressing fingertip's color red
                for tip_id, (_, _, is_pressed) in zip(self.fingertips, hand_info):
                    if is_pressed:
                        x = int(hand_landmarks.landmark[tip_id].x * frame.shape[1])
                        y = int(hand_landmarks.landmark[tip_id].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

        return frame, hands_data, thresholds


class PianoKey:
    def __init__(self, x, y, width, height, note):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.note = note
        self.is_playing = False
        self.last_played = 0
        self.cooldown = 0.1


class ARPiano:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.setup_audio()
        self.setup_piano_keys()
        self.keyboard_img = None
        self.transform_matrix = None

    def setup_audio(self):
        mixer.init(44100, -16, 2, 2048)
        self.sounds = {}
        notes = ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6']
        for note in notes:
            try:
                self.sounds[note] = mixer.Sound(f'notes/{note}.wav')
            except:
                print(f"Warning: Could not find {note}.wav")

    def setup_piano_keys(self):
        self.keys = []
        key_width = 100
        key_height = 300
        start_x = 0
        start_y = 0
        notes = ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6']
        for i, note in enumerate(notes):
            x = start_x + (i * key_width)
            key = PianoKey(x, start_y, key_width, key_height, note)
            self.keys.append(key)

    def load_keyboard(self):
        if self.keyboard_img is None:
            try:
                self.keyboard_img = cv2.imread('keyboard.png')
                if self.keyboard_img is None:
                    raise FileNotFoundError("Could not find keyboard.png")
                self.keyboard_img = cv2.resize(self.keyboard_img, (800, 300))
            except Exception as e:
                print(f"Error: {e}")
                return None
        return self.keyboard_img

    def transform_keyboard(self, frame, corners):
        keyboard = self.load_keyboard()
        if keyboard is None or corners is None:
            return frame

        try:
            h, w = keyboard.shape[:2]
            keyboard_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            doc_pts = np.float32(corners)
            self.transform_matrix = cv2.getPerspectiveTransform(keyboard_pts, doc_pts)
            warped = cv2.warpPerspective(keyboard, self.transform_matrix, 
                                       (frame.shape[1], frame.shape[0]))
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
            result = frame.copy()
            result[mask] = cv2.addWeighted(frame, 0.3, warped, 0.85, 0)[mask]
            return result
        except Exception as e:
            print(f"Error: {e}")
            return frame

    def check_key_press(self, x, y, is_pressed):
        if self.transform_matrix is None:
            return
            
        try:
            finger_pos = np.array([[[x, y]]], dtype=np.float32)
            inv_matrix = np.linalg.inv(self.transform_matrix)
            transformed = cv2.perspectiveTransform(finger_pos, inv_matrix)
            x_transformed = transformed[0][0][0]
            
            if is_pressed:
                for key in self.keys:
                    if key.x <= x_transformed <= key.x + key.width:
                        current_time = time.time()
                        if not key.is_playing and \
                           (current_time - key.last_played) > key.cooldown:
                            if key.note in self.sounds:
                                self.sounds[key.note].play()
                            key.is_playing = True
                            key.last_played = current_time
                        return
            else:
                for key in self.keys:
                    key.is_playing = False
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("Initializing AR Piano...")

    prog = PianoApp()
    prog.run()

if __name__ == "__main__":
    main()
