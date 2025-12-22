from flask import Flask, render_template, Response
from ar_engine import (
    generate_frame,
    next_style,
    prev_style,
    get_face_shape
)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def video_stream():
    while True:
        frame = generate_frame()
        if frame is None:
            break
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video")
def video():
    return Response(video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/next")
def next_glasses():
    next_style()
    return ("", 204)

@app.route("/prev")
def prev_glasses():
    prev_style()
    return ("", 204)

@app.route("/face-shape")
def face_shape():
    return get_face_shape()

if __name__ == "__main__":
    app.run(debug=True)
