# Smart Lock System - Hand Gesture Password

A Python application that uses OpenCV and MediaPipe to create a smart password locking system based on hand gestures. The system detects hand gestures through your webcam (including your mobile phone as a camera) and uses finger counting as a password mechanism.

---

## Features

- 🔐 **Hand Gesture Recognition**: Uses MediaPipe to detect and track hand landmarks
- 👆 **Finger Counting**: Counts the number of fingers shown using a custom-trained model or geometric rules
- 🔢 **Password System**: Maps finger counts to numbers (1 finger = 1, 2 fingers = 2, etc.)
- 🎯 **Sequence Matching**: Stores a predefined password and checks user input against it
- ⏱️ **Gesture Delay**: Prevents duplicate detection with configurable delay between gestures
- 🖥️ **GUI Interface**: Modern Tkinter-based user interface with real-time status updates
- 📱 **Mobile Camera Support**: Use your phone as a webcam via IP camera apps
- 🧠 **Custom Training**: Place labeled images (0.jpeg, 1.jpeg, ..., 5.jpeg) in an `images/` folder to train the system on your own hand
- 🔄 **Auto Reset**: Automatically resets input after password verification
- 🛡️ **Start Input Button**: Prevents accidental input by only collecting gestures after you press "Start Input"

---

## Requirements

- Python 3.7 or higher
- Webcam **or** mobile phone (as IP camera)
- Good lighting for hand detection

---

## No Webcam? Use Your Mobile Phone as a Webcam (DroidCam/IP Webcam)

If your PC does **not** have a webcam, you can use your mobile phone as a webcam using apps like **DroidCam** or **IP Webcam**.

### Using DroidCam (Recommended)
1. **Install DroidCam** on your phone:
   - [Android](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
   - [iOS](https://apps.apple.com/us/app/droidcam-webcam-for-pc/id1510258102)
2. **Install the DroidCam Client** on your PC from [here](https://www.dev47apps.com/).
3. **Connect your phone and PC to the same WiFi network** (or use USB).
4. **Start the DroidCam app** on your phone and note the IP address and port.
5. **Start the DroidCam Client** on your PC and enter the IP address and port.
6. **Start the video**. DroidCam will create a virtual webcam on your PC (usually Camera 1 or 2).
7. **In the Python app**, set the camera index to 1 or 2:
   ```python
   self.camera = cv2.VideoCapture(1)  # or 2
   ```

### Using IP Webcam (Android)
1. **Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)** on your Android phone.
2. **Connect your phone and PC to the same WiFi network.**
3. **Start the app** and note the video stream URL (e.g., `http://192.168.1.13:4747/video`).
4. **In the Python app**, set the camera source to the URL:
   ```python
   self.camera = cv2.VideoCapture("http://YOUR_PHONE_IP:PORT/video")
   ```

---

## Installation

1. **Clone or download the project files**
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Custom Training (Optional, for best accuracy)

1. **Create a folder named `images` in the project directory.**
2. **Add one clear JPEG image for each finger count:**
   - `0.jpeg` (fist)
   - `1.jpeg` (one finger)
   - `2.jpeg` (two fingers)
   - `3.jpeg` (three fingers)
   - `4.jpeg` (four fingers)
   - `5.jpeg` (open palm)
3. **On first run, the app will automatically train a model** using these images. If you want to retrain, delete `finger_classifier.pkl` and restart the app.

---

## Usage

1. **Run the application**:
   ```bash
   python smart_lock_system.py
   ```
2. **Click "Start Camera"** to begin hand detection.
3. **Click "Start Input"** when you are ready to show your gesture password.
4. **Show hand gestures to input the password** (default: 2 fingers → 1 finger → 5 fingers).
5. **Wait 1.5 seconds between gestures.**
6. **The system will automatically check after 3 gestures.**
7. **Use "Reset Input"** to clear the current input sequence and try again.

---

## How It Works

- **Hand Detection:** Uses MediaPipe Hands for real-time hand landmark detection.
- **Finger Counting:** Uses a custom-trained classifier (if images are provided) or a geometric method.
- **Password System:** Compares your gesture sequence to the stored password.
- **Visual Feedback:** Shows hand landmarks, finger count, and status in both the GUI and camera feed.
- **Start Input Button:** Prevents accidental input by only collecting gestures after you press "Start Input".

---

## Troubleshooting

- **OpenCV GUI Error:** If you see an error about `cv2.imshow` not being implemented, make sure you have installed `opencv-contrib-python` (not `opencv-python-headless`).
- **NumPy Version Error:** If you see errors about NumPy version, install a compatible version:
  ```bash
  pip install "numpy>=1.25.2,<2.0.0"
  ```
- **Hand Not Detected:** Ensure good lighting, clear background, and your hand is centered in the frame.
- **Mobile Camera Not Working:** Double-check the stream URL and that both devices are on the same network.

---

## Customization

- **Change the Password:** Edit the `self.password` variable in `smart_lock_system.py`.
- **Change Camera Source:** Edit the `cv2.VideoCapture` line in `smart_lock_system.py`.
- **Retrain the Model:** Add new images to `images/`, delete `finger_classifier.pkl`, and restart the app.

---

## License

This project is open source and available under the MIT License.

---

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system. 
=======
# Smart-Locking-System-OpenCV
 Python application that uses OpenCV and MediaPipe to create a smart password locking system based on hand gestures. The system detects hand gestures through your webcam (including your mobile phone as a camera) and uses finger counting as a password mechanism.
>>>>>>> fbfd3b76531848541591d9ccf252ffbe461f6dcb
