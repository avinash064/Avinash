# import os
# import argparse
# import yaml
# import time
# from glob import glob

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# import timm
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter

# # ────────────────────────────────────────────────────────
# # DATASET DEFINITION
# # ────────────────────────────────────────────────────────
# class FrameDataset(Dataset):
#     """
#     Dataset for loading extracted frames from Anim400K
#     Directory structure: root/{video_id}/frame_0001.jpg
#     """
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = sorted(glob(os.path.join(root_dir, '*', '*.jpg')))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# # ────────────────────────────────────────────────────────
# # MAE MODEL WRAPPER
# # ────────────────────────────────────────────────────────
# class MAEWrapper(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = nn.Sequential(
#             nn.Linear(encoder.embed_dim, encoder.embed_dim),
#             nn.ReLU(),
#             nn.Linear(encoder.embed_dim, encoder.patch_embed.proj.weight.shape[1] * encoder.patch_embed.proj.weight.shape[2])
#         )

#     def forward(self, x):
#         latent = self.encoder(x)
#         output = self.decoder(latent)
#         return output

# # ────────────────────────────────────────────────────────
# # TRAINING FUNCTION
# # ────────────────────────────────────────────────────────
# def train(args):
#     # TensorBoard
#     writer = SummaryWriter(log_dir=args.ckpt_dir)

#     # Data transforms
#     transform = transforms.Compose([
#         transforms.Resize((args.image_size, args.image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     # Dataset and loader
#     dataset = FrameDataset(args.data_root, transform)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

#     # Model and optimizer
#     model = MAEWrapper(timm.create_model('vit_base_patch16_224', pretrained=False)).cuda()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
#     scaler = GradScaler()
#     criterion = nn.MSELoss()

#     # Resume from checkpoint
#     start_epoch = 0
#     if args.resume:
#         ckpt = torch.load(args.resume)
#         model.load_state_dict(ckpt['model'])
#         optimizer.load_state_dict(ckpt['optimizer'])
#         start_epoch = ckpt['epoch'] + 1
#         print(f"[✔] Resumed from epoch {start_epoch}")

#     # Training loop
#     for epoch in range(start_epoch, args.epochs):
#         model.train()
#         epoch_loss = 0

#         for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
#             images = batch.cuda(non_blocking=True)
#             optimizer.zero_grad()

#             with autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, images)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(dataloader)
#         writer.add_scalar('Loss/train', avg_loss, epoch)
#         print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

#         # Save checkpoint
#         os.makedirs(args.ckpt_dir, exist_ok=True)
#         ckpt_path = os.path.join(args.ckpt_dir, f"mae_epoch_{epoch+1}.pt")
#         torch.save({
#             'epoch': epoch,
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }, ckpt_path)

# # ────────────────────────────────────────────────────────
# # ARGUMENT PARSER / CONFIGURATION
# # ────────────────────────────────────────────────────────
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, help='Path to YAML config')
#     parser.add_argument('--data_root', type=str, default='datasets/anim400k/frames')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--image_size', type=int, default=224)
#     parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
#     parser.add_argument('--resume', type=str, default=None)
#     args = parser.parse_args()

#     # Override args with YAML if provided
#     if args.config:
#         with open(args.config, 'r') as f:
#             yaml_args = yaml.safe_load(f)
#         for key, val in yaml_args.items():
#             setattr(args, key, val)

#     return args

# # ────────────────────────────────────────────────────────
# # MAIN ENTRY
# # ────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     args = parse_args()
#     train(args)
# #     print("All folders validated successfully.")
# #     print("\n📂 Step 2: Extract frames from videos"   )
# import os
# import argparse
# import yaml
# import time
# from glob import glob

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# import timm
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.utils as vutils

# # ────────────────────────────────────────────────────────
# # DATASET DEFINITION
# # ────────────────────────────────────────────────────────
# class FrameDataset(Dataset):
#     """
#     Dataset for loading extracted frames from Anim400K
#     Directory structure: root/{video_id}/frame_0001.jpg
#     """
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = sorted(glob(os.path.join(root_dir, '*', '*.jpg')))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# # ────────────────────────────────────────────────────────
# # MAE MODEL WRAPPER
# # ────────────────────────────────────────────────────────
# class MAEWrapper(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         embed_dim = encoder.embed_dim
#         num_patches = encoder.patch_embed.num_patches
#         self.decoder = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 3 * 224 * 224 // num_patches)
#         )

#     def forward(self, x):
#         latent = self.encoder(x)
#         output = self.decoder(latent)
#         return output

# # ────────────────────────────────────────────────────────
# # TRAINING FUNCTION
# # ────────────────────────────────────────────────────────
# def train(args):
#     # TensorBoard
#     writer = SummaryWriter(log_dir=args.ckpt_dir)

#     # Data transforms with augmentation
#     transform = transforms.Compose([
#         transforms.Resize((args.image_size, args.image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     # Dataset and loader
#     dataset = FrameDataset(args.data_root, transform)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

#     # Model and optimizer
#     model = MAEWrapper(timm.create_model('vit_base_patch16_224', pretrained=False)).cuda()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scaler = GradScaler()
#     criterion = nn.MSELoss()

#     # Resume from checkpoint
#     start_epoch = 0
#     if args.resume:
#         ckpt = torch.load(args.resume)
#         model.load_state_dict(ckpt['model'])
#         optimizer.load_state_dict(ckpt['optimizer'])
#         start_epoch = ckpt['epoch'] + 1
#         print(f"[✔] Resumed from epoch {start_epoch}")

#     # Training loop
#     for epoch in range(start_epoch, args.epochs):
#         model.train()
#         epoch_loss = 0

#         for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
#             images = batch.cuda(non_blocking=True)
#             optimizer.zero_grad()

#             with autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, images)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             epoch_loss += loss.item()

#             # Log example reconstructions
#             if step == 0:
#                 with torch.no_grad():
#                     recon_images = outputs[:4].reshape(-1, 3, args.image_size, args.image_size)
#                     writer.add_images('Reconstructions', recon_images, epoch)
#                     writer.add_images('Originals', images[:4], epoch)

#         avg_loss = epoch_loss / len(dataloader)
#         writer.add_scalar('Loss/train', avg_loss, epoch)
#         print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

#         # Save checkpoint
#         os.makedirs(args.ckpt_dir, exist_ok=True)
#         ckpt_path = os.path.join(args.ckpt_dir, f"mae_epoch_{epoch+1}.pt")
#         torch.save({
#             'epoch': epoch,
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }, ckpt_path)

# # ────────────────────────────────────────────────────────
# # ARGUMENT PARSER / CONFIGURATION
# # ────────────────────────────────────────────────────────
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, help='Path to YAML config')
#     parser.add_argument('--data_root', type=str, default='datasets/anim400k/frames')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--image_size', type=int, default=224)
#     parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
#     parser.add_argument('--resume', type=str, default=None)
#     args = parser.parse_args()

#     # Override args with YAML if provided
#     if args.config:
#         with open(args.config, 'r') as f:
#             yaml_args = yaml.safe_load(f)
#         for key, val in yaml_args.items():
#             setattr(args, key, val)

#     return args

# # ────────────────────────────────────────────────────────
# # MAIN ENTRY
# # ────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     args = parse_args()
#     train(args)
# #     print("All folders validated successfully.")
# #     print("\n📂 Step 2: Extract frames from videos"   )
import os
import argparse
import yaml
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# ────────────────────────────────────────────────────────
# DATASET DEFINITION
# ────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    """
    Dataset for loading extracted frames from Anim400K
    Directory structure: root/{video_id}/frame_0001.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob(os.path.join(root_dir, '*', '*.jpg')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# ────────────────────────────────────────────────────────
# MAE MODEL WRAPPER
# ────────────────────────────────────────────────────────
class MAEWrapper(nn.Module):
    def __init__(self, image_size=224, embed_dim=768):
        super().__init__()
        # Create ViT without classification head (num_classes=0)
        self.encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0
        )
        self.image_size = image_size
        self.embed_dim = embed_dim
        # Decoder: reconstruct full image
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3 * image_size * image_size),
        )

    def forward(self, x):
        # x: [B,3,H,W]
        # encoder returns latent embeddings [B, embed_dim]
        latent = self.encoder(x)
        recon = self.decoder(latent)  # [B, 3*H*W]
        B = x.size(0)
        return recon.view(B, 3, self.image_size, self.image_size)

# ────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ────────────────────────────────────────────────────────
def train(args):
    writer = SummaryWriter(log_dir=args.ckpt_dir)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    dataset = FrameDataset(args.data_root, transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    model = MAEWrapper(args.image_size, embed_dim=768).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    criterion = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        for step, imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            imgs = imgs.cuda(non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Save ckpt
        os.makedirs(args.ckpt_dir, exist_ok=True)
        path = os.path.join(args.ckpt_dir, f"mae_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)

# ────────────────────────────────────────────────────────
# ARG PARSER
# ────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--data_root', type=str, default='datasets/anim400k/frames')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        for k, v in cfg.items(): setattr(args, k, v)
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)
