import os
import zipfile
import cv2
from tqdm import tqdm
from glob import glob

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = "datasets"
ANIM_DIR = os.path.join(BASE_DIR, "anim400k")
ANITA_ZIP = os.path.join(BASE_DIR, "anitadataset.zip")
ANITA_DIR = os.path.join(BASE_DIR, "anitadataset")

VIDEO_DIR = os.path.join(ANIM_DIR, "video_clips")
AUDIO_DIR = os.path.join(ANIM_DIR, "audio_clips")
FRAME_DIR = os.path.join(ANIM_DIR, "frames")
ANNOTATION_DIR = os.path.join(ANIM_DIR, "annotations")
CHARACTER_PIC_DIR = os.path.join(ANIM_DIR, "character_pics")

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
def extract_zip(zip_path, dest_dir):
    print(f"[⇪] Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(dest_dir)

def extract_frames_from_videos(video_dir, output_dir, fps=4, img_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    videos = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"[🎞️] Found {len(videos)} videos in {video_dir}")

    for vid_path in tqdm(videos, desc="🎬 Extracting Frames"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_subdir = os.path.join(output_dir, vid_name)
        os.makedirs(out_subdir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        fps_actual = cap.get(cv2.CAP_PROP_FPS) or fps
        interval = max(int(fps_actual // fps), 1)

        frame_id, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % interval == 0:
                frame = cv2.resize(frame, img_size)
                frame_path = os.path.join(out_subdir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1
            frame_id += 1
        cap.release()

# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
def main():
    print("\n📂 Step 1: Validate folders")
    print(f"📁 Videos:      {VIDEO_DIR}")
    print(f"🎵 Audio:       {AUDIO_DIR}")
    print(f"📸 Characters:  {CHARACTER_PIC_DIR}")
    print(f"🧾 Annotations: {ANNOTATION_DIR}")
    print(f"🖼️ Frame Out:   {FRAME_DIR}")

    if os.path.exists(ANITA_ZIP) and not os.path.exists(ANITA_DIR):
        print("\n📦 Step 2: Extract AnitaDataset")
        extract_zip(ANITA_ZIP, ANITA_DIR)
    else:
        print(f"[✔] AnitaDataset already extracted or not found.")

    print("\n🎞️ Step 3: Extract video frames")
    extract_frames_from_videos(VIDEO_DIR, FRAME_DIR, fps=4, img_size=(512, 512))

    print("\n✅ Dataset preparation complete.")

if __name__ == "__main__":
    main()
