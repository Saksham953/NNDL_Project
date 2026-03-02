import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from models.generator import Generator
from models.keypoint_detector import KeypointDetector
from models.motion_network import MotionNet
from utils.preprocessing import extract_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
if os.path.exists("generator.pth"):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

kp_detector = KeypointDetector().to(device)
if os.path.exists("kp_detector.pth"):
    kp_detector.load_state_dict(torch.load("kp_detector.pth", map_location=device))
kp_detector.eval()

motion_net = MotionNet().to(device)
if os.path.exists("motion_net.pth"):
    motion_net.load_state_dict(torch.load("motion_net.pth", map_location=device))
motion_net.eval()

def animate(source_image_path, driving_video_path):
    print(f"Animating {source_image_path} with motion from {driving_video_path}...")
    
    # Preprocess source image
    source_img = cv2.imread(source_image_path)
    if source_img is None:
        raise FileNotFoundError(f"Failed to load image from {source_image_path}")
    source_img = cv2.resize(source_img, (256, 256))
    source_img = source_img / 255.0
    source = torch.tensor(source_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Preprocess driving video
    driving_frames = extract_frames(driving_video_path)
    if len(driving_frames) == 0:
        raise ValueError(f"No frames found in driving video: {driving_video_path}")
    
    output_frames = []

    with torch.no_grad():
        kp_s = kp_detector(source)
        kp_s_flat = kp_s.view(1, -1)

        hidden = None
        motion_sequence = []
        for i in range(len(driving_frames)):
            driving_tensor = torch.tensor(driving_frames[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            kp_d = kp_detector(driving_tensor)
            kp_d_flat = kp_d.view(1, -1)

            motion, hidden = motion_net(kp_s_flat, kp_d_flat, hidden)
            motion_sequence.append(motion)
            
            valid_seq = motion_sequence[-20:]
            motion_seq_tensor = torch.stack(valid_seq, dim=1)
            
            pred = generator(source, motion_seq_tensor)
            
            pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()
            pred = np.clip(pred, 0, 1)
            pred = (pred * 255).astype(np.uint8)

            output_frames.append(pred)

    # Save video
    h, w, _ = output_frames[0].shape
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    for frame in output_frames:
        out.write(frame)

    out.release()
    print("Animation saved to output.mp4 successfully.")

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    print("Welcome to the Advanced Animation System!")
    
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    print("Opening file dialog to select the static source image...")
    source_image_path = filedialog.askopenfilename(
        title="Select Static Source Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not source_image_path:
        print("Error: No source image selected. Exiting.")
        sys.exit(1)

    print("Opening file dialog to select the driving video...")
    driving_video_path = filedialog.askopenfilename(
        title="Select Driving Video",
        filetypes=[("Video files", "*.mp4 *.mpg *.avi *.mov *.mkv")]
    )
    
    if not driving_video_path:
        print("Error: No driving video selected. Exiting.")
        sys.exit(1)
        
    if not os.path.exists(source_image_path):
        print(f"Error: Source image not found at {source_image_path}")
        sys.exit(1)
        
    if not os.path.exists(driving_video_path):
        print(f"Error: Driving video not found at {driving_video_path}")
        sys.exit(1)

    animate(source_image_path, driving_video_path)