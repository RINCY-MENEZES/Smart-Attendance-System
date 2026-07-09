# Smart Attendance System

## Overview
A Python-based facial recognition attendance system that automates attendance using OpenCV and the `face_recognition` library. The project supports Hikvision CCTV (RTSP) streams for real-time attendance monitoring.

## Features
- Real-time face detection and recognition
- Automatic attendance logging (CSV)
- Webcam support
- Hikvision CCTV (RTSP) integration
- Timestamp-based attendance recording

## Technologies
- Python
- OpenCV
- face_recognition
- NumPy

## Project Structure
```
Documentation/      -> Project report & paper
ImagesAttendance/   -> Face dataset
face_attendance.py  -> Main program
face_encodings.py   -> Face encoding generation
mark_attendance.py  -> Attendance logging
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python face_attendance.py
```

## Documentation

The project report and published paper are available in the `Documentation` folder.
