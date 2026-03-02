import cv2

def extract_frames(video_path, size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video at {video_path}")
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, size)
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    return frames