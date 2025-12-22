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

# Load glasses for different face shapes
glasses_map = {
    "Round": cv2.imread("assets/glasses_rectangular.png", cv2.IMREAD_UNCHANGED),
    "Square": cv2.imread("assets/glasses_round.png", cv2.IMREAD_UNCHANGED),
    "Oval": cv2.imread("assets/glasses_oval.png", cv2.IMREAD_UNCHANGED)
}

# Buffers for smoothing
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)
width_buffer = deque(maxlen=5)
shape_buffer = deque(maxlen=7)

def overlay_transparent(background, overlay, x, y, overlay_size):
    overlay = cv2.resize(overlay, overlay_size)

    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if ox2 <= ox1 or oy2 <= oy1:
        return background

    alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha * overlay[oy1:oy2, ox1:ox2, c] +
            (1 - alpha) * background[y1:y2, x1:x2, c]
        )

    return background

def classify_face_shape(width, height):
    ratio = height / width
    if ratio < 1.2:
        return "Round"
    elif ratio < 1.35:
        return "Square"
    else:
        return "Oval"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Eye landmarks
        x1, y1 = int(lm[33].x * w), int(lm[33].y * h)
        x2, y2 = int(lm[263].x * w), int(lm[263].y * h)

        eye_width = int(math.dist((x1, y1), (x2, y2)) * 1.9)
        glasses_height = int(eye_width * 0.5)

        x = x1 - int(eye_width * 0.25)
        y = y1 - int(glasses_height * 0.5)

        x_buffer.append(x)
        y_buffer.append(y)
        width_buffer.append(eye_width)

        smooth_x = int(np.mean(x_buffer))
        smooth_y = int(np.mean(y_buffer))
        smooth_width = int(np.mean(width_buffer))
        smooth_height = int(smooth_width * 0.5)

        face_width = math.dist(
            (lm[234].x, lm[234].y),
            (lm[454].x, lm[454].y)
        ) * w

        face_height = math.dist(
            (lm[10].x, lm[10].y),
            (lm[152].x, lm[152].y)
        ) * h

        shape = classify_face_shape(face_width, face_height)
        shape_buffer.append(shape)
        stable_shape = max(set(shape_buffer), key=shape_buffer.count)

        glasses = glasses_map[stable_shape]

        frame = overlay_transparent(
            frame,
            glasses,
            smooth_x,
            smooth_y,
            (smooth_width, smooth_height)
        )

        cv2.putText(frame, f"Face Shape: {stable_shape}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3)

    cv2.imshow("Day 6 - Smart Glasses Recommendation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
