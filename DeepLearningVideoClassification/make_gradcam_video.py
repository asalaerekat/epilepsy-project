import os
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.cm as cm

import torchvision.transforms as T
from dataset import VideoDataset, VideoAugmentation
from models import get_model
from gradcam_3d import GradCAM3D

MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3,1,1)

def denorm_frame(x_chw: torch.Tensor) -> np.ndarray:
    # x_chw: (3,H,W) normalized float
    img = (x_chw.cpu() * STD + MEAN).clamp(0, 1)  # (3,H,W)
    img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)  # (H,W,3)
    return img

def overlay(frame_uint8: np.ndarray, cam_hw: np.ndarray, alpha=0.45) -> np.ndarray:
    # cam_hw in [0,1]
    heat = (cm.jet(cam_hw)[...,:3] * 255).astype(np.uint8)
    out = (frame_uint8 * (1 - alpha) + heat * alpha).astype(np.uint8)
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # deterministic val transform (important for interpretability)
    val_tf = T.Compose([
        T.Resize((128, 171)),
        T.CenterCrop((112,112)),
        T.ToTensor(),
        T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
    ])
    val_transform = VideoAugmentation(val_tf)

    video_dir = "/path/to/videos"
    num_frames = 16
    ds = VideoDataset(video_dir, num_frames=num_frames, transform=val_transform)

    idx = 0  # choose which video
    x, y = ds[idx]                 # x: (C,T,H,W)
    x = x.unsqueeze(0).to(device)  # (1,C,T,H,W)

    # load model + checkpoint
    model = get_model("resnet3d", pretrained=True, num_classes=1).to(device)
    ckpt_path = "/path/to/best_foldX.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # choose target layer
    m = model.module if hasattr(model, "module") else model
    target_layer = m.layer4[-1].conv2

    cam_fn = GradCAM3D(model, target_layer)

    cam = cam_fn(x)[0]  # (T,H,W) for the single sample

    # build overlay frames
    x0 = x[0].detach().cpu()  # (C,T,H,W) normalized
    frames = []
    for t in range(x0.shape[1]):
        frame = denorm_frame(x0[:, t])        # (H,W,3) uint8
        cam_t = cam[t].cpu().numpy()          # (H,W) in [0,1]
        frames.append(overlay(frame, cam_t))

    out_gif = "gradcam.gif"
    imageio.mimsave(out_gif, frames, fps=6)
    print("Saved:", out_gif)
    cam_fn.close()

if __name__ == "__main__":
    main()