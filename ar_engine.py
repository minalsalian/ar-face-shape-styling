import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

# Glasses styles per face shape
glasses_styles = {
    "Round": [
        cv2.imread("assets/rect_1.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/rect_2.png", cv2.IMREAD_UNCHANGED)
    ],
    "Square": [
        cv2.imread("assets/round_1.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/round_2.png", cv2.IMREAD_UNCHANGED)
    ],
    "Oval": [
        cv2.imread("assets/oval_1.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("assets/oval_2.png", cv2.IMREAD_UNCHANGED)
    ]
}

style_index = 0

# Smoothing buffers
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)
width_buffer = deque(maxlen=5)
shape_buffer = deque(maxlen=7)

def next_style():
    global style_index
    style_index = (style_index + 1) % 2

def prev_style():
    global style_index
    style_index = (style_index - 1) % 2

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

def classify_face_shape(w, h):
    r = h / w
    if r < 1.2:
        return "Round"
    elif r < 1.35:
        return "Square"
    else:
        return "Oval"

def generate_frame():
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # Eye landmarks
        x1, y1 = int(lm[33].x * w), int(lm[33].y * h)
        x2, y2 = int(lm[263].x * w), int(lm[263].y * h)

        # Eye center (KEY FIX)
        eye_center_x = (x1 + x2) // 2
        eye_center_y = (y1 + y2) // 2

        # Glasses size
        eye_width = int(math.dist((x1, y1), (x2, y2)) * 1.9)
        glasses_height = int(eye_width * 0.5)

        # Correct placement (slightly BELOW eye line)
        x = eye_center_x - eye_width // 2
        y = eye_center_y - glasses_height // 2 + int(glasses_height * 0.05)

        # Smoothing
        x_buffer.append(x)
        y_buffer.append(y)
        width_buffer.append(eye_width)

        sx = int(np.mean(x_buffer))
        sy = int(np.mean(y_buffer))
        sw = int(np.mean(width_buffer))
        sh = int(sw * 0.5)

        # Face shape
        fw = math.dist((lm[234].x, lm[234].y), (lm[454].x, lm[454].y)) * w
        fh = math.dist((lm[10].x, lm[10].y), (lm[152].x, lm[152].y)) * h

        shape = classify_face_shape(fw, fh)
        shape_buffer.append(shape)
        stable_shape = max(set(shape_buffer), key=shape_buffer.count)

        glasses = glasses_styles[stable_shape][style_index]

        frame = overlay_transparent(frame, glasses, sx, sy, (sw, sh))

        cv2.putText(
            frame,
            f"{stable_shape} Face",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

    _, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes()
