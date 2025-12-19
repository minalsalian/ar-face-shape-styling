import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

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
        landmarks = results.multi_face_landmarks[0].landmark

        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]

        face_width = distance(left_cheek, right_cheek) * w
        face_height = distance(forehead, chin) * h

        face_shape = classify_face_shape(face_width, face_height)

        cv2.putText(frame, f"Face Shape: {face_shape}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3)

        cv2.putText(frame, f"Ratio: {face_height/face_width:.2f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    cv2.imshow("Day 3 - Face Shape Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
