import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class VideoAugmentation:
    """
    Clip-consistent video augmentation.

    Input:
      video tensor with shape (C, T, H, W), usually uint8 [0,255]

    Output:
      video tensor with shape (C, T, crop_h, crop_w), float normalized

    Important: all random spatial transforms are sampled ONCE per clip and
    applied consistently to every frame. This preserves temporal coherence.
    """

    def __init__(
        self,
        is_train: bool,
        resize_size: Tuple[int, int] = (128, 171),
        crop_size: Tuple[int, int] = (112, 112),
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        crop_ratio: Tuple[float, float] = (0.9, 1.1),
        hflip_prob: float = 0.5,
        mean: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989),
    ):
        self.is_train = is_train
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.hflip_prob = hflip_prob
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 4:
            raise ValueError(f"Expected video shape (C,T,H,W), got {tuple(video.shape)}")

        # Convert (C,T,H,W) -> (T,C,H,W), float [0,1]
        video = video.permute(1, 0, 2, 3).contiguous()
        if video.dtype != torch.float32:
            video = video.to(torch.float32)
        if video.max() > 1.0:
            video = video / 255.0

        # Resize every frame deterministically.
        frames = [TF.resize(frame, self.resize_size, antialias=True) for frame in video]

        if self.is_train:
            # Sample one random crop and one flip decision per clip.
            i, j, h, w = T.RandomResizedCrop.get_params(
                frames[0], scale=self.crop_scale, ratio=self.crop_ratio
            )
            frames = [
                TF.resized_crop(
                    frame,
                    top=i,
                    left=j,
                    height=h,
                    width=w,
                    size=self.crop_size,
                    antialias=True,
                )
                for frame in frames
            ]
            if random.random() < self.hflip_prob:
                frames = [TF.hflip(frame) for frame in frames]
        else:
            frames = [TF.center_crop(frame, self.crop_size) for frame in frames]

        video = torch.stack(frames, dim=0)  # (T,C,H,W)

        # Normalize. Broadcasting over T.
        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        video = (video - mean) / std

        return video.permute(1, 0, 2, 3).contiguous()  # (C,T,H,W)


class VideoDataset(Dataset):
    """
    Loads videos named *_<LABEL>.mp4.

    Label mapping:
      PNEE/FDS -> 0
      GTC/FocalBilateralTC/ES -> 1

    sample_mode:
      - "sparse": old behavior; sample num_frames uniformly across the full usable window.
      - "random_clip": training behavior; choose a random local clip each __getitem__.
      - "center_clip": deterministic validation behavior; choose the center local clip.

    For external testing, use get_multiclip() to sample many deterministic local clips
    and aggregate predictions at the video level in train.py.
    """

    VALID_ENDINGS = ("PNEE.mp4", "FDS.mp4", "GTC.mp4", "FocalBilateralTC.mp4", "ES.mp4")

    def __init__(
        self,
        video_dir,
        num_frames=16,
        transform=None,
        file_list=None,
        sample_mode="center_clip",
        clip_fps=4.0,
        start_trim_sec=118.0,
        end_trim_sec=120.0,
    ):
        self.video_dir = video_dir
        if file_list is None:
            self.video_files = [f for f in os.listdir(video_dir) if f.endswith(self.VALID_ENDINGS)]
        else:
            self.video_files = file_list
        self.video_files = list(self.video_files)
        print("VideoDataset: found", len(self.video_files), "files")

        self.num_frames = int(num_frames)
        self.transform = transform
        self.sample_mode = sample_mode
        self.clip_fps = float(clip_fps)
        self.start_trim_sec = float(start_trim_sec)
        self.end_trim_sec = float(end_trim_sec)

        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if self.clip_fps <= 0:
            raise ValueError("clip_fps must be > 0")
        if self.sample_mode not in {"sparse", "random_clip", "center_clip"}:
            raise ValueError(
                "sample_mode must be one of: 'sparse', 'random_clip', 'center_clip'"
            )

    def __len__(self):
        return len(self.video_files)

    def _get_label(self, fn):
        lbl = os.path.splitext(fn)[0].split("_")[-1]
        if lbl in ("PNEE", "FDS"):
            return 0
        if lbl in ("GTC", "FocalBilateralTC", "ES"):
            return 1
        raise ValueError("Unknown label in " + fn)

    def _video_path(self, fn):
        return os.path.join(self.video_dir, fn)

    def _get_usable_window(self, fn):
        path = self._video_path(fn)
        try:
            timestamps, _ = io.read_video_timestamps(path, pts_unit="sec")
        except Exception as exc:
            raise RuntimeError(f"Failed reading timestamps for {fn}") from exc

        if len(timestamps) == 0:
            raise ValueError(f"{fn} has no readable video frames")

        duration_sec = float(timestamps[-1])
        usable_start = self.start_trim_sec
        usable_end = duration_sec - self.end_trim_sec
        if usable_end <= usable_start:
            raise ValueError(
                f"{fn} too short after cut: duration={duration_sec:.2f}s, "
                f"needs > {self.start_trim_sec + self.end_trim_sec:.2f}s"
            )
        return path, duration_sec, usable_start, usable_end

    @property
    def local_clip_duration_sec(self):
        return float(self.num_frames) / float(self.clip_fps)

    def _choose_clip_window(self, usable_start, usable_end, mode):
        usable_duration = usable_end - usable_start
        clip_duration = min(self.local_clip_duration_sec, usable_duration)
        max_start = usable_end - clip_duration

        if mode == "random_clip" and max_start > usable_start:
            clip_start = random.uniform(usable_start, max_start)
        elif mode == "center_clip" and max_start > usable_start:
            clip_start = usable_start + 0.5 * (max_start - usable_start)
        else:
            clip_start = usable_start

        clip_end = clip_start + clip_duration
        return float(clip_start), float(clip_end)

    def _read_video_window(self, path, fn, start_sec, end_sec):
        vid, _, _ = io.read_video(
            path,
            start_pts=float(start_sec),
            end_pts=float(end_sec),
            pts_unit="sec",
        )  # (T,H,W,C)
        if vid.numel() == 0:
            raise ValueError(
                f"{fn} produced empty clip after window [{start_sec:.2f}, {end_sec:.2f}] sec"
            )

        vid = vid.permute(0, 3, 1, 2).contiguous()  # (T,C,H,W)

        # Ensure exactly 3 channels.
        if vid.shape[1] == 1:
            vid = vid.repeat(1, 3, 1, 1)
        elif vid.shape[1] > 3:
            vid = vid[:, :3, :, :]

        return vid

    def _sample_or_pad_frames(self, vid):
        # vid: (T,C,H,W). Return (C,num_frames,H,W).
        t_now = vid.shape[0]
        if t_now >= self.num_frames:
            idxs = torch.linspace(0, t_now - 1, steps=self.num_frames).long()
            vid = vid[idxs]
        else:
            pad_count = self.num_frames - t_now
            pad = vid[-1:].repeat(pad_count, 1, 1, 1)
            vid = torch.cat([vid, pad], dim=0)

        return vid.permute(1, 0, 2, 3).contiguous()

    def _load_clip(self, fn, start_sec, end_sec):
        path = self._video_path(fn)
        vid = self._read_video_window(path, fn, start_sec, end_sec)
        vid = self._sample_or_pad_frames(vid)
        if self.transform:
            vid = self.transform(vid)
        return vid

    def _load_sparse_full_window(self, fn, usable_start, usable_end):
        path = self._video_path(fn)
        vid = self._read_video_window(path, fn, usable_start, usable_end)
        vid = self._sample_or_pad_frames(vid)
        if self.transform:
            vid = self.transform(vid)
        return vid

    def __getitem__(self, idx):
        fn = self.video_files[idx]
        label = self._get_label(fn)
        _, _, usable_start, usable_end = self._get_usable_window(fn)

        if self.sample_mode == "sparse":
            vid = self._load_sparse_full_window(fn, usable_start, usable_end)
        else:
            clip_start, clip_end = self._choose_clip_window(
                usable_start, usable_end, self.sample_mode
            )
            vid = self._load_clip(fn, clip_start, clip_end)

        return vid, label

    def get_multiclip(self, fn, num_clips=20):
        """
        Deterministically sample multiple local clips across the usable video window.

        Returns:
          clips: Tensor (num_actual_clips, C, T, H, W)
          label: int
          rows: list of metadata rows for clip-level output
        """
        label = self._get_label(fn)
        _, duration_sec, usable_start, usable_end = self._get_usable_window(fn)

        num_clips = int(num_clips)
        if num_clips < 1:
            raise ValueError("num_clips must be >= 1")

        usable_duration = usable_end - usable_start
        clip_duration = min(self.local_clip_duration_sec, usable_duration)
        max_start = usable_end - clip_duration

        if num_clips == 1 or max_start <= usable_start:
            starts = [usable_start + 0.5 * max(0.0, max_start - usable_start)]
        else:
            starts = torch.linspace(usable_start, max_start, steps=num_clips).tolist()

        clips = []
        rows = []
        for clip_index, start_sec in enumerate(starts):
            end_sec = float(start_sec) + clip_duration
            clip = self._load_clip(fn, float(start_sec), float(end_sec))
            clips.append(clip)
            rows.append(
                {
                    "filename": fn,
                    "true_label": int(label),
                    "clip_index": int(clip_index),
                    "clip_start_sec": float(start_sec),
                    "clip_end_sec": float(end_sec),
                    "video_duration_sec": float(duration_sec),
                    "usable_start_sec": float(usable_start),
                    "usable_end_sec": float(usable_end),
                    "sample_mode": "multi_clip",
                }
            )

        return torch.stack(clips, dim=0), label, rows
