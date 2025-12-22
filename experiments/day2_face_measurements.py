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

# Helper function to calculate distance
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

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

        # Select landmarks
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]

        # Calculate measurements
        face_width = distance(left_cheek, right_cheek) * w
        face_height = distance(forehead, chin) * h

        # Display values
        cv2.putText(frame, f"Face Width: {int(face_width)}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Face Height: {int(face_height)}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Draw points
        for point in [234, 454, 10, 152]:
            x = int(landmarks[point].x * w)
            y = int(landmarks[point].y * h)
            cv2.circle(frame, (x, y), 5, (0,0,255), -1)

    cv2.imshow("Day 2 - Face Measurements", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
