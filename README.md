# AR Face Shape Styling System 👓✨

A real-time augmented reality (AR) web application that detects facial landmarks, classifies face shape, and overlays personalized glasses using computer vision.

---

## 🚀 Features
- Real-time face landmark detection (MediaPipe)
- Face width & height measurement
- Rule-based face shape classification (Round / Square / Oval)
- Stable AR glasses overlay with smoothing
- Automatic glasses recommendation based on face shape
- Multiple glasses styles with user switching
- Web-based interface using Flask

---

## 🧠 Face Shape Logic
Face shape is determined using the ratio:
face_ratio = face_height / face_width

| Ratio Range | Face Shape |
|------------|-----------|
| < 1.2 | Round |
| 1.2 – 1.35 | Square |
| > 1.35 | Oval |

---

## 🛠️ Tech Stack
- Python 3.10
- OpenCV
- MediaPipe
- NumPy
- Flask
- HTML (Jinja templates)

---

## 📂 Project Structure
ar-face-shape/
├── app.py
├── ar_engine.py
├── day1_face_landmarks.py
├── day2_face_measurements.py
├── day3_face_shape.py
├── day4_ar_glasses.py
├── day5_stable_ar_glasses.py
├── day6_face_shape_recommendation.py
├── day7_multi_style_switching.py
├── assets/
├── templates/
└── README.md


---

## ▶️ How to Run (Web App)

 1. Install dependencies
py -3.10 -m pip install opencv-python mediapipe flask numpy imutils

2. Run the app
py -3.10 app.py

3. Open browser
http://127.0.0.1:5000

🎮 Controls
N → Next glasses style
P → Previous glasses style
Q → Quit application

🌱 Future Enhancements
Hairstyle AR
Emotion-based recommendations
Mobile app (Android / iOS)
ML-based face shape classification
Online deployment


