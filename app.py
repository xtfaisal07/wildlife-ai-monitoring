from flask import Flask, render_template, Response, request, send_file
import cv2
import os
import sqlite3
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

# ---------------- CONFIG ----------------
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# GLOBAL PATHS
last_video_path = ""
last_image_path = ""


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  detected TEXT,
                  time TEXT)''')
    conn.commit()
    conn.close()

init_db()


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/video_upload")
def video_upload():
    return render_template("video.html")


# ---------------- IMAGE DETECTION ----------------
@app.route("/detect", methods=["POST"])
def detect():
    global last_image_path

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    results = model(filepath)

    detected_objects = []
    alert = False

    if not results or len(results[0].boxes) == 0:
        return render_template("result.html",
                               image=filepath,
                               detected="No objects detected",
                               alert=False)

    annotated = results[0].plot()

    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + file.filename)
    cv2.imwrite(output_path, annotated)

    # ✅ SAVE IMAGE PATH
    last_image_path = output_path

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)

            if label == "person":
                alert = True

            detected_objects.append(f"{label} ({conf:.2f})")

    detected_text = ", ".join(set(detected_objects))

    # SAVE TO DB
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO detections (filename, detected, time) VALUES (?, ?, ?)",
              (file.filename, detected_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           image=output_path,
                           detected=detected_text,
                           alert=alert)


# ---------------- DOWNLOAD IMAGE ----------------
@app.route("/download_image")
def download_image():
    global last_image_path

    if last_image_path == "":
        return "No image available"

    return send_file(last_image_path, as_attachment=True)


# ---------------- VIDEO PROCESSING ----------------
@app.route("/process_video", methods=["POST"])
def process_video():
    global last_video_path

    file = request.files["video"]

    if file.filename == "":
        return "No video selected"

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output_" + file.filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    detected_objects = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                detected_objects.add(label)

    cap.release()
    out.release()

    # SAVE VIDEO PATH
    last_video_path = output_path

    detected_text = ", ".join(detected_objects) if detected_objects else "No objects detected"

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO detections (filename, detected, time) VALUES (?, ?, ?)",
              (file.filename, detected_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           video=output_path,
                           detected=detected_text)


# ---------------- DOWNLOAD VIDEO ----------------
@app.route("/download_video")
def download_video():
    global last_video_path

    if last_video_path == "":
        return "No video available"

    return send_file(last_video_path, as_attachment=True)


# ---------------- LIVE CAMERA ----------------
def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY id DESC")
    data = c.fetchall()

    total = len(data)

    counts = {}
    for row in data:
        labels = row[2].split(",")
        for label in labels:
            name = label.split("(")[0].strip()
            counts[name] = counts.get(name, 0) + 1

    conn.close()

    return render_template("dashboard.html",
                           data=data,
                           total=total,
                           counts=counts)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
