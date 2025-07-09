# import os
# import tarfile
# import requests
# from tqdm import tqdm
# import subprocess

# ANIM400K_FILES = {
#     "video_clips": "https://s3.us-west-1.wasabisys.com/anim400k/anim400k_video_clips.tar.gz",
#     "audio_clips": "https://s3.us-west-1.wasabisys.com/anim400k/anim400k_audio_clips.tar.gz",
#     "character_pics": "https://s3.us-west-1.wasabisys.com/anim400k/anim400k_character_pics.tar.gz",
#     "annotations": "https://s3.us-west-1.wasabisys.com/anim400k/anim400k_annotations_official_splits_v1.json"
# }

# ANITA_GDRIVE_FILE_ID = "1ctfD0sMpT2pVutJUOlyEYKhAxufMYmZ_"

# DATASET_ROOT = "datasets"
# ANIM_ROOT = os.path.join(DATASET_ROOT, "anim400k")
# ANITA_PATH = os.path.join(DATASET_ROOT, "anitadataset.zip")

# def download_file(url, out_path):
#     if os.path.exists(out_path):
#         print(f"[✔] {out_path} already exists.")
#         return
#     print(f"[↓] Downloading {url}...")
#     response = requests.get(url, stream=True)
#     total = int(response.headers.get('content-length', 0))
#     with open(out_path, 'wb') as file, tqdm(
#         desc=out_path,
#         total=total,
#         unit='B',
#         unit_scale=True,
#         unit_divisor=1024,
#     ) as bar:
#         for data in response.iter_content(chunk_size=1024):
#             size = file.write(data)
#             bar.update(size)

# def extract_tar(file_path, extract_to):
#     if not os.path.exists(extract_to):
#         os.makedirs(extract_to)
#     print(f"[⇪] Extracting {file_path}...")
#     with tarfile.open(file_path, 'r:gz') as tar:
#         tar.extractall(path=extract_to)

# def extract_zip(zip_path, extract_to):
#     import zipfile
#     print(f"[⇪] Extracting {zip_path}...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)

# def download_anim400k():
#     os.makedirs(ANIM_ROOT, exist_ok=True)
#     # Annotations
#     download_file(ANIM400K_FILES["annotations"], os.path.join(ANIM_ROOT, "splits.json"))
#     # Tarballs
#     for key, url in ANIM400K_FILES.items():
#         if key == "annotations":
#             continue
#         tar_path = os.path.join(ANIM_ROOT, f"{key}.tar.gz")
#         download_file(url, tar_path)
#         extract_to = os.path.join(ANIM_ROOT, key)
#         extract_tar(tar_path, extract_to)

# def download_anitadataset():
#     print("[↓] Downloading AnitaDataset from Google Drive...")
#     if not os.path.exists(ANITA_PATH):
#         # Uses gdown
#         os.system(f"gdown --id {ANITA_GDRIVE_FILE_ID} -O {ANITA_PATH}")
#     extract_zip(ANITA_PATH, os.path.join(DATASET_ROOT, "anitadataset"))

# def extract_frames(video_dir, output_dir, fps=4, img_size=(512, 512)):
#     import cv2
#     from glob import glob

#     os.makedirs(output_dir, exist_ok=True)
#     video_files = sorted(glob(os.path.join(video_dir, "*.mp4")))

#     for video_path in tqdm(video_files, desc="Extracting frames"):
#         vid_name = os.path.splitext(os.path.basename(video_path))[0]
#         out_dir = os.path.join(output_dir, vid_name)
#         os.makedirs(out_dir, exist_ok=True)

#         cap = cv2.VideoCapture(video_path)
#         frame_rate = cap.get(cv2.CAP_PROP_FPS)
#         frame_interval = int(frame_rate // fps) if frame_rate > 0 else 1

#         count, saved = 0, 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if count % frame_interval == 0:
#                 frame = cv2.resize(frame, img_size)
#                 out_path = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
#                 cv2.imwrite(out_path, frame)
#                 saved += 1
#             count += 1
#         cap.release()

# def run_mae():
#     print("[🧠] Starting MAE pretraining...")
#     subprocess.run(["python", "scripts/train_mae.py"])

# if __name__ == "__main__":
#     print("🚀 Step 1: Downloading Anim400K")
#     download_anim400k()

#     print("\n🎨 Step 2: Downloading AnitaDataset")
#     download_anitadataset()

#     print("\n🖼️ Step 3: Extracting frames from Anim400K")
#     extract_frames(
#         video_dir=os.path.join(ANIM_ROOT, "video_clips"),
#         output_dir=os.path.join(ANIM_ROOT, "frames"),
#         fps=4,
#         img_size=(512, 512)
#     )

#     print("\n🧠 Step 4: Launching MAE Pretraining")
#     run_mae()
#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import tarfile
import requests
import zipfile
import subprocess
from tqdm import tqdm

# ────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────
ANIM400K_FILES = {
    "video_clips":   "",  # User will download manually
    "audio_clips":   "https://s3.us-west-1.wasabisys.com/anim400k/anim400k_audio_clips.tar.gz",
    "character_pics":"https://s3.us-west-1.wasabisys.com/anim400k/anim400k_character_pics.tar.gz",
    "annotations":   "https://s3.us-west-1.wasabisys.com/anim400k/anim400k_annotations_official_splits_v1.json"
}

ANITA_GDRIVE_FILE_ID = "1ctfD0sMpT2pVutJUOlyEYKhAxufMYmZ_"

DATASET_ROOT = "datasets"
ANIM_ROOT    = os.path.join(DATASET_ROOT, "anim400k")
ANITA_ZIP    = os.path.join(DATASET_ROOT, "anitadataset.zip")
ANITA_DIR    = os.path.join(DATASET_ROOT, "anitadataset")

# ────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────
def download_file(url, out_path):
    print(f"[↓] Downloading {url.split('?')[0]} → {out_path}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(out_path, 'wb') as f, tqdm(
        desc=os.path.basename(out_path),
        total=total, unit='B', unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def is_gzip(path):
    with open(path, 'rb') as f:
        magic = f.read(2)
    return magic == b'\x1f\x8b'

def extract_tar_safe(tar_path, dest_dir):
    print(f"[⇪] Extracting {tar_path} → {dest_dir}")
    if not is_gzip(tar_path):
        print(f"[!] {tar_path} is not a valid gzip. Skipping extraction.")
        return False
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"[!] Extraction failed: {e}")
        return False

def extract_zip(zip_path, dest_dir):
    print(f"[⇪] Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(dest_dir)

# ────────────────────────────────────────────────────────
# STEP FUNCTIONS
# ────────────────────────────────────────────────────────
def download_anim400k():
    os.makedirs(ANIM_ROOT, exist_ok=True)

    # annotations
    ann_path = os.path.join(ANIM_ROOT, "splits.json")
    if not os.path.exists(ann_path):
        download_file(ANIM400K_FILES["annotations"], ann_path)
    else:
        print(f"[✔] {ann_path} already exists.")

    # media tarballs
    for key, url in ANIM400K_FILES.items():
        if key == "annotations" or url.strip() == "":
            continue
        tar_path = os.path.join(ANIM_ROOT, f"{key}.tar.gz")
        extract_to = os.path.join(ANIM_ROOT, key)
        os.makedirs(extract_to, exist_ok=True)

        if not os.path.exists(tar_path):
            download_file(url, tar_path)

        if not extract_tar_safe(tar_path, extract_to):
            print(f"[✗] Failed to extract {key}. Please manually check {tar_path}.")

def download_anitadataset():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    print(f"[↓] Downloading AnitaDataset via gdown")
    if not os.path.exists(ANITA_ZIP):
        subprocess.run([
            "gdown", "--id", ANITA_GDRIVE_FILE_ID,
            "-O", ANITA_ZIP
        ], check=True)
    else:
        print(f"[✔] {ANITA_ZIP} already exists.")
    if not os.path.exists(ANITA_DIR):
        extract_zip(ANITA_ZIP, ANITA_DIR)
    else:
        print(f"[✔] {ANITA_DIR} already exists.")

def extract_frames(video_dir, output_dir, fps=4, img_size=(512, 512)):
    import cv2
    from glob import glob

    os.makedirs(output_dir, exist_ok=True)
    videos = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"[🖼️] Extracting frames from {len(videos)} videos → {output_dir}")

    for vid in tqdm(videos, desc="Videos"):
        name = os.path.splitext(os.path.basename(vid))[0]
        out_sub = os.path.join(output_dir, name)
        os.makedirs(out_sub, exist_ok=True)

        cap = cv2.VideoCapture(vid)
        fps_orig = cap.get(cv2.CAP_PROP_FPS) or fps
        interval = max(int(fps_orig // fps), 1)

        count = saved = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                frame = cv2.resize(frame, img_size)
                cv2.imwrite(os.path.join(out_sub, f"frame_{saved:04d}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()

def run_mae_pretraining():
    print("[🧠] Launching MAE pretraining (scripts/train_mae.py)")
    subprocess.run(["python3", "scripts/train_mae.py"], check=True)

# ────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────
def main():
    print("\n🚀 Step 1: Download & prepare Anim400K")
    download_anim400k()

    print("\n🎨 Step 2: Download & extract AnitaDataset")
    download_anitadataset()

    print("\n🖼️ Step 3: Extract frames from Anim400K videos")
    extract_frames(
        video_dir = os.path.join(ANIM_ROOT, "video_clips_1"),  # Change if extracted to custom dir
        output_dir = os.path.join(ANIM_ROOT, "frames"),
        fps = 4,
        img_size = (512, 512)
    )

    print("\n🧠 Step 4: Start MAE pretraining")
    run_mae_pretraining()

    print("\n✅ Day 1 & 2 pipeline complete!")

if __name__ == "__main__":
    main()
 

# This script is designed to be run in a Python environment with the necessary libraries installed.
# Ensure you have `requests`, `tqdm`, `opencv-python`, and `gdown` installed.
# You can install them via pip:
# pip install requests tqdm opencv-python gdown     
