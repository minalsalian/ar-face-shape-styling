import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# -------------------------------
# MediaPipe Setup
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

# -------------------------------
# Global face shape (for UI badge)
# -------------------------------
current_face_shape = "Detecting..."

# -------------------------------
# Glasses recommendation mapping
# -------------------------------
glasses_styles = {
    "Round": [
       
        cv2.imread("assets/oval.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/geometric.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/cat-eye.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/square.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/windsor.png", cv2.IMREAD_UNCHANGED)
    ],
    "Square": [
        cv2.imread("assets/round.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/rimless.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/oval.png", cv2.IMREAD_UNCHANGED)
    ],
    "Oval": [
      
        cv2.imread("assets/oval.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/cat-eye.png", cv2.IMREAD_UNCHANGED)

    ],
    "Heart": [
        
        cv2.imread("assets/oval.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/cat-eye.png", cv2.IMREAD_UNCHANGED)
    ],
    "Diamond": [
        cv2.imread("assets/rimless.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/round.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/oval.png", cv2.IMREAD_UNCHANGED),
    ],
    "Oblong": [
        cv2.imread("assets/rimless.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/round.png", cv2.IMREAD_UNCHANGED)
    ]
}

style_index = 0

# -------------------------------
# Smoothing buffers
# -------------------------------
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)
width_buffer = deque(maxlen=5)
shape_buffer = deque(maxlen=7)

# -------------------------------
# Style controls (buttons)
# -------------------------------
def next_style():
    global style_index
    style_index = (style_index + 1) % 2

def prev_style():
    global style_index
    style_index = (style_index - 1) % 2

def get_face_shape():
    return current_face_shape

# -------------------------------
# Overlay PNG helper
# -------------------------------
def overlay_transparent(bg, overlay, x, y, size):
    overlay = cv2.resize(overlay, size)

    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if ox2 <= ox1 or oy2 <= oy1:
        return bg

    alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0

    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            alpha * overlay[oy1:oy2, ox1:ox2, c] +
            (1 - alpha) * bg[y1:y2, x1:x2, c]
        )

    return bg

# -------------------------------
# 6 Face Shape Classification (TUNED)
# -------------------------------
def classify_face_shape(face_h, face_w, jaw_w, cheek_w, forehead_w):
    ratio = face_h / face_w

    jaw_r = jaw_w / face_w
    cheek_r = cheek_w / face_w
    forehead_r = forehead_w / face_w

    # 1️⃣ Oblong (Rectangle) – long face, straight sides
    if ratio > 1.5 and abs(jaw_r - cheek_r) < 0.06:
        return "Oblong"

    # 2️⃣ Heart – wide forehead, narrow jaw
    if forehead_r > cheek_r + 0.03 and jaw_r < cheek_r - 0.04:
        return "Heart"

    # 3️⃣ Diamond – rare, sharp cheekbones
    if (
        cheek_r > forehead_r + 0.05 and
        cheek_r > jaw_r + 0.05 and
        ratio > 1.3
    ):
        return "Diamond"

    # 4️⃣ Square – strong jaw, similar width & height
    if ratio < 1.25 and abs(jaw_r - cheek_r) < 0.05:
        return "Square"

    # 5️⃣ Round – soft curves, equal width & height
    # Round – soft curves, cheeks not dominant
    if (
        ratio < 1.25 and
        cheek_r < forehead_r + 0.03 and
        cheek_r < jaw_r + 0.03
    ):
        return "Round"

    # 6️⃣ Oval – balanced (default)
    return "Oval"


# -------------------------------
# Frame generator (Flask stream)
# -------------------------------
def generate_frame():
    global current_face_shape

    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Eye landmarks
        x1, y1 = int(lm[33].x * w), int(lm[33].y * h)
        x2, y2 = int(lm[263].x * w), int(lm[263].y * h)

        eye_center_x = (x1 + x2) // 2
        eye_center_y = (y1 + y2) // 2

        eye_width = int(math.dist((x1, y1), (x2, y2)) * 1.9)
        glasses_height = int(eye_width * 0.5)

        x = eye_center_x - eye_width // 2
        y = eye_center_y - glasses_height // 2 + int(glasses_height * 0.05)

        x_buffer.append(x)
        y_buffer.append(y)
        width_buffer.append(eye_width)

        sx = int(np.mean(x_buffer))
        sy = int(np.mean(y_buffer))
        sw = int(np.mean(width_buffer))
        sh = int(sw * 0.5)

        # Facial measurements
        face_width = math.dist((lm[234].x, lm[234].y), (lm[454].x, lm[454].y)) * w
        face_height = math.dist((lm[10].x, lm[10].y), (lm[152].x, lm[152].y)) * h
        forehead_width = math.dist((lm[67].x, lm[67].y), (lm[297].x, lm[297].y)) * w
        cheekbone_width = math.dist((lm[123].x, lm[123].y), (lm[352].x, lm[352].y)) * w
        jaw_width = math.dist((lm[172].x, lm[172].y), (lm[397].x, lm[397].y)) * w

        shape = classify_face_shape(
            face_height,
            face_width,
            jaw_width,
            cheekbone_width,
            forehead_width
        )

        shape_buffer.append(shape)
        stable_shape = max(set(shape_buffer), key=shape_buffer.count)
        current_face_shape = stable_shape

        glasses = glasses_styles[stable_shape][style_index]
        frame = overlay_transparent(frame, glasses, sx, sy, (sw, sh))

    _, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes()
