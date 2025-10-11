import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import os

def make_video_from_frames(frame_dir, output_path, fps=10):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    if not frame_files:
        print("No frames found!")
        return
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for fname in frame_files:
        frame = cv2.imread(os.path.join(frame_dir, fname))
        video.write(frame)
    video.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    make_video_from_frames('log/sample/test', 'output.mp4', fps=10)