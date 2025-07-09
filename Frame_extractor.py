import os
import cv2
from glob import glob
from tqdm import tqdm

# ────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────
BASE_DIR = "datasets/anim400k"
VIDEO_DIR = os.path.join(BASE_DIR, "video_clips", "anim400k_video_clips")
FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "frames")
TARGET_FPS = 4
IMG_SIZE = (512, 512)

# ────────────────────────────────────────────────────────
# FRAME EXTRACTION FUNCTION
# ────────────────────────────────────────────────────────
def extract_frames_from_videos(video_root, output_root, fps=4, img_size=(512, 512)):
    print(f"[🖼️] Scanning video files in {video_root} ...")
    
    video_files = glob(os.path.join(video_root, "*", "*.mp4"))  # e.g., 00/video.mp4
    print(f"[✔] Found {len(video_files)} video files.")

    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_root, video_name)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        interval = max(int(orig_fps // fps), 1)

        frame_idx = 0
        saved_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frame = cv2.resize(frame, img_size)
                out_path = os.path.join(output_dir, f"frame_{saved_idx:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_idx += 1
            frame_idx += 1

        cap.release()

# ────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────
def main():
    print("🚀 Starting Anim400K frame extraction")
    extract_frames_from_videos(
        video_root=VIDEO_DIR,
        output_root=FRAME_OUTPUT_DIR,
        fps=TARGET_FPS,
        img_size=IMG_SIZE
    )
    print(f"\n✅ All frames extracted to: {FRAME_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
