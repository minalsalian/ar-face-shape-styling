import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

# Load glasses image (must be PNG with transparency)
glasses = cv2.imread("assets/glasses.png", cv2.IMREAD_UNCHANGED)

def overlay_transparent(background, overlay, x, y, overlay_size):
    overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    return background

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = landmarks[33]
        right_eye = landmarks[263]

        # Convert to pixel coordinates
        x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
        x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

        eye_width = int(math.dist((x1, y1), (x2, y2)) * 1.9)
        glasses_height = int(eye_width * 0.5)

        x = x1 - int(eye_width * 0.25)
        y = y1 - int(glasses_height * 0.5)

        frame = overlay_transparent(
            frame,
            glasses,
            x,
            y,
            (eye_width, glasses_height)
        )

    cv2.imshow("Day 4 - AR Glasses", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
