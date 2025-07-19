import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from collections import deque
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class SmartLockSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.password = [2, 1, 5]
        self.input_sequence = []
        self.last_gesture_time = 0
        self.gesture_delay = 1.5
        self.collecting_input = False
        self.access_granted = False
        self.is_running = False
        self.camera = None
        self.model_file = 'finger_classifier.pkl'
        self.classifier = None
        self.train_classifier_from_images()
        self.load_classifier()
        self.create_gui()
    def train_classifier_from_images(self):
        if os.path.exists(self.model_file):
            return
        image_dir = 'images'
        labels = [0, 1, 2, 3, 4, 5]
        X, y = [], []
        static_hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        for label in labels:
            img_path = os.path.join(image_dir, f'{label}.jpeg')
            image = cv2.imread(img_path)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = static_hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_vec = []
                    for lm in hand_landmarks.landmark:
                        landmark_vec.extend([lm.x, lm.y, lm.z])
                    X.append(landmark_vec)
                    y.append(label)
        if len(X) >= 6:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)
            joblib.dump(clf, self.model_file)
    def load_classifier(self):
        if os.path.exists(self.model_file):
            self.classifier = joblib.load(self.model_file)
        else:
            self.classifier = None
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Smart Lock System - Hand Gesture Password")
        self.root.geometry("400x500")
        self.root.configure(bg='#2c3e50')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='white', background='#2c3e50')
        style.configure('Status.TLabel', font=('Arial', 12), foreground='white', background='#2c3e50')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#bdc3c7', background='#2c3e50')
        title_label = ttk.Label(self.root, text="🔐 Smart Lock System", style='Title.TLabel')
        title_label.pack(pady=20)
        status_frame = tk.Frame(self.root, bg='#2c3e50')
        status_frame.pack(pady=20)
        self.status_label = ttk.Label(status_frame, text="🔒 LOCKED", style='Status.TLabel')
        self.status_label.pack()
        self.gesture_label = ttk.Label(status_frame, text="Show hand gesture...", style='Info.TLabel')
        self.gesture_label.pack(pady=10)
        self.sequence_label = ttk.Label(status_frame, text="Input: []", style='Info.TLabel')
        self.sequence_label.pack(pady=5)
        password_text = f"Password: {self.password}"
        self.password_label = ttk.Label(status_frame, text=password_text, style='Info.TLabel')
        self.password_label.pack(pady=5)
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=30)
        self.start_button = tk.Button(
            button_frame,
            text="Start Camera",
            command=self.toggle_camera,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=10
        )
        self.start_button.pack(pady=10)
        self.reset_button = tk.Button(
            button_frame,
            text="Reset Input",
            command=self.reset_input,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=10
        )
        self.reset_button.pack(pady=10)
        self.start_input_button = tk.Button(
            button_frame,
            text="Start Input",
            command=self.start_input,
            bg='#2980b9',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=10
        )
        self.start_input_button.pack(pady=10)
        instructions = """
Instructions:
1. Click 'Start Camera' to begin
2. Click 'Start Input' when ready to show gestures
3. Show hand gestures to input password
4. Password: 2 fingers → 1 finger → 5 fingers
5. Wait 1.5 seconds between gestures
6. System will automatically check after 3 gestures
        """
        instruction_label = ttk.Label(self.root, text=instructions, style='Info.TLabel', justify='left')
        instruction_label.pack(pady=20)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    def start_input(self):
        self.collecting_input = True
        self.input_sequence = []
        self.sequence_label.config(text="Input: []")
        self.status_label.config(text="🔒 LOCKED", foreground='white')
        self.gesture_label.config(text="Show hand gesture...")
    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    def start_camera(self):
        self.camera = cv2.VideoCapture("http://192.168.1.13:4747/video")
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        self.is_running = True
        self.start_button.config(text="Stop Camera", bg='#e74c3c')
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.start_button.config(text="Start Camera", bg='#27ae60')
        cv2.destroyAllWindows()
    def count_fingers(self, hand_landmarks):
        if not hand_landmarks:
            return 0
        if self.classifier is not None:
            landmark_vec = []
            for lm in hand_landmarks.landmark:
                landmark_vec.extend([lm.x, lm.y, lm.z])
            X = np.array(landmark_vec).reshape(1, -1)
            pred = self.classifier.predict(X)
            return int(pred[0])
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        landmarks = np.array(landmarks)
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 7, 11, 15, 19]
        finger_count = 0
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:
                if landmarks[tip][0] < landmarks[pip][0]:
                    finger_count += 1
            else:
                if landmarks[tip][1] < landmarks[pip][1]:
                    finger_count += 1
        return finger_count
    def process_gesture(self, finger_count):
        current_time = time.time()
        if not self.collecting_input:
            return
        if current_time - self.last_gesture_time < self.gesture_delay:
            return
        self.input_sequence.append(finger_count)
        self.last_gesture_time = current_time
        self.sequence_label.config(text=f"Input: {self.input_sequence}")
        if len(self.input_sequence) >= len(self.password):
            self.check_password()
    def check_password(self):
        if self.input_sequence == self.password:
            self.access_granted = True
            self.status_label.config(text="🔓 ACCESS GRANTED")
            self.status_label.config(foreground='#27ae60')
            messagebox.showinfo("Success", "Access Granted! Welcome!")
        else:
            self.access_granted = False
            self.status_label.config(text="❌ ACCESS DENIED")
            self.status_label.config(foreground='#e74c3c')
            messagebox.showerror("Access Denied", "Incorrect password sequence!")
        self.reset_input()
    def reset_input(self):
        self.input_sequence = []
        self.sequence_label.config(text="Input: []")
        self.collecting_input = False
        if not self.access_granted:
            self.status_label.config(text="🔒 LOCKED")
            self.status_label.config(foreground='white')
        self.gesture_label.config(text="Show hand gesture...")
    def camera_loop(self):
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    finger_count = self.count_fingers(hand_landmarks)
                    self.root.after(0, lambda: self.gesture_label.config(text=f"Current: {finger_count} fingers"))
                    self.process_gesture(finger_count)
                    cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.root.after(0, lambda: self.gesture_label.config(text="Show hand gesture..."))
            status_text = "ACCESS GRANTED" if self.access_granted else "LOCKED"
            status_color = (0, 255, 0) if self.access_granted else (0, 0, 255)
            cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            sequence_text = f"Input: {self.input_sequence}"
            cv2.putText(frame, sequence_text, (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Smart Lock System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    smart_lock = SmartLockSystem()
    smart_lock.run() 