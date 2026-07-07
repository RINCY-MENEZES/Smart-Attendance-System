import cv2
import numpy as np
import face_recognition
import threading
import time
from face_encodings import encodeListKnown, classNames
from mark_attendance import markAttendance

# ============================================================
# 1️⃣ Hikvision RTSP Configuration (Main Stream recommended)
# ============================================================
# Format: rtsp://username:password@IP:554/Streaming/Channels/101
RTSP_URL = "rtsp://admin:root@100@192.168.11.7/Streaming/Channels/101"

# ============================================================
# 2️⃣ Threaded Frame Grabber Class — For Real-Time Processing
# ============================================================
class FrameGrabber:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print("❌ Cannot open Hikvision stream. Check URL / credentials.")
            exit()

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # Try to read FPS (fallback to 15)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        print(f"🎥 Connected to Hikvision stream (~{self.fps:.1f} FPS)")

        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            # Read the latest frame
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


# ============================================================
# 3️⃣ Start Stream Thread
# ============================================================
stream = FrameGrabber(RTSP_URL)
time.sleep(1.0)  # Give the stream time to warm up

print("✅ Starting Real-Time Face Recognition — Press 'Q' to Quit")

# ============================================================
# 4️⃣ Main Processing Loop
# ============================================================
while True:
    frame = stream.read()
    if frame is None:
        continue

    # Convert frame for recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize slightly for speed, but not too much (CCTV = far faces)
    small_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # Detect faces
    face_locations = face_recognition.face_locations(small_rgb, model='hog')
    face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        # Scale back since we used 0.5
        y1, x2, y2, x1 = int(y1 * 2), int(x2 * 2), int(y2 * 2), int(x1 * 2)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)
            markAttendance(name)
        else:
            name = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show video in real-time
    cv2.imshow("Hikvision Face Recognition - Real Time", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()