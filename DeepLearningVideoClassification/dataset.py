import os
import torch
from torch.utils.data import Dataset
import torchvision.io as io
from torchvision.transforms.functional import to_pil_image

class VideoAugmentation:
    """
    Wraps a torchvision.transforms pipeline that expects a PIL image.
    Applies frame-wise, returns a Tensor of shape (C, T, H, W).
    """
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, video: torch.Tensor):
        # video: (C, T, H, W)
        frames = []
        C, T, H, W = video.shape
        for t in range(T):
            frame = video[:, t, :, :]          # (C, H, W)
            pil   = to_pil_image(frame)        # to PIL
            aug   = self.image_transform(pil)  # apply your transform
            # Tensor output: (C, H, W)
            if not isinstance(aug, torch.Tensor):
                aug = aug.to(torch.float32)
            frames.append(aug)
        # stack back → (T, C, H, W) then permute (C, T, H, W)
        stacked = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return stacked

class VideoDataset(Dataset):
    """
    Loads videos named *_<LABEL>.mp4, cuts start/end,
    samples self.num_frames uniformly, then applies transform.
    Returns (video_tensor, label_int).
    """
    def __init__(self, video_dir, num_frames=16, transform=None, file_list=None):
        self.video_dir   = video_dir
        endings = ("PNEE.mp4", "FDS.mp4", "GTC.mp4", "FocalBilateralTC.mp4", "ES.mp4")
        if file_list is None:
            self.video_files = [f for f in os.listdir(video_dir) if f.endswith(endings)]
        else:
            self.video_files = file_list
        print("VideoDataset: found", len(self.video_files), "files")
        self.num_frames = num_frames
        self.transform  = transform

    def __len__(self):
        return len(self.video_files)

    def _get_label(self, fn):
        lbl = os.path.splitext(fn)[0].split("_")[-1]
        if lbl in ("PNEE", "FDS"): return 0
        if lbl in ("GTC", "FocalBilateralTC", "ES"): return 1
        raise ValueError("Unknown label in "+fn)

    def __getitem__(self, idx):
        fn    = self.video_files[idx]
        path  = os.path.join(self.video_dir, fn)
        label = self._get_label(fn)

        vid, _, info = io.read_video(path, pts_unit="sec")  # (T,H,W,C)
        fps = info.get("video_fps", 25)
        vid = vid.permute(0,3,1,2)  #  (T,C,H,W)

        # ensure 3 channels
        if vid.shape[1]==1:
            vid = vid.repeat(1,3,1,1)
        elif vid.shape[1]>3:
            vid = vid[:,:3]

        # cut start/end
        start = int(118*fps)
        end   = int(120*fps)
        Ttot  = vid.shape[0]
        if Ttot > start+end:
            vid = vid[start:Ttot-end]
        else:
            raise ValueError(f"{fn} too short after cut")

        # sample/pad to fixed length
        Tnow = vid.shape[0]
        if Tnow >= self.num_frames:
            idxs = torch.linspace(0, Tnow-1, steps=self.num_frames).long()
            vid = vid[idxs]
        else:
            last = vid[-1:]
            while vid.shape[0] < self.num_frames:
                vid = torch.cat([vid, last], dim=0)
            vid = vid[:self.num_frames]

        # → (C,T,H,W)
        vid = vid.permute(1,0,2,3)

        if self.transform:
            vid = self.transform(vid)

        return vid, label
