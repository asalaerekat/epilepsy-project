import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoAugmentation, VideoDataset  # noqa: E402
from gradcam_3d import GradCAM3D  # noqa: E402
from models import get_model  # noqa: E402


MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D Grad-CAM overlay GIF/MP4 for a video sample.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory of labeled videos.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="gradcam_outputs", help="Output directory.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index in VideoDataset.")
    parser.add_argument("--filename", type=str, default=None, help="Optional exact filename to select.")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames sampled per clip.")
    parser.add_argument("--model", type=str, default="resnet3d", choices=["resnet3d", "vmz"])
    parser.add_argument("--pretrained", action="store_true", help="Initialize model with pretrained weights.")
    parser.add_argument("--num_classes", type=int, default=1, help="Model output classes (1 for binary logit head).")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--target_layer", type=str, default=None,
                        help="Layer path (e.g., layer4.1.conv2). Default: model.layer4[-1].conv2")
    parser.add_argument("--class_idx", type=int, default=None,
                        help="For binary head (N,1): 1=seizure CAM (logit), 0=nonseizure CAM (-logit).")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay blend factor.")
    parser.add_argument("--cam_threshold", type=float, default=0.0,
                        help="Suppress CAM values below this threshold in [0,1].")
    parser.add_argument("--cam_gamma", type=float, default=1.0,
                        help="Power applied to CAM before overlay (>=1 sharpens focus).")
    parser.add_argument("--fps", type=int, default=6, help="GIF/MP4 frames per second.")
    parser.add_argument("--save_format", type=str, default="both", choices=["gif", "mp4", "both"])
    parser.add_argument("--colormap", type=str, default="jet", help="Matplotlib colormap name.")
    parser.add_argument("--use_data_parallel", action="store_true", help="Wrap model in DataParallel if possible.")
    return parser.parse_args()


def get_val_transform():
    val_tf = T.Compose([
        T.Resize((128, 171)),
        T.CenterCrop((112, 112)),
        T.ToTensor(),
        T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
    ])
    return VideoAugmentation(val_tf)


def denorm_frame(x_chw: torch.Tensor) -> np.ndarray:
    img = (x_chw.cpu() * STD + MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def overlay_frame(
    frame_uint8: np.ndarray,
    cam_hw: np.ndarray,
    alpha: float,
    colormap: str,
    cam_threshold: float = 0.0,
    cam_gamma: float = 1.0,
) -> np.ndarray:
    cam_vis = np.clip(cam_hw, 0.0, 1.0)
    if cam_threshold > 0.0:
        cam_vis = np.where(cam_vis >= cam_threshold, cam_vis, 0.0)
    if cam_gamma != 1.0:
        cam_vis = np.power(cam_vis, cam_gamma)

    heat = (cm.get_cmap(colormap)(cam_vis)[..., :3] * 255.0).astype(np.uint8)
    alpha_map = (alpha * cam_vis)[..., None].astype(np.float32)
    out = frame_uint8.astype(np.float32) * (1.0 - alpha_map) + heat.astype(np.float32) * alpha_map
    return np.clip(out, 0, 255).astype(np.uint8)


def _is_int_token(token: str) -> bool:
    return token.lstrip("-").isdigit()


def resolve_layer(model: torch.nn.Module, layer_path: str = None) -> torch.nn.Module:
    base_model = model.module if hasattr(model, "module") else model
    if not layer_path:
        return base_model.layer4[-1].conv2

    current = base_model
    for token in layer_path.split("."):
        if _is_int_token(token):
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def strip_module_prefix(state_dict):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def add_module_prefix(state_dict):
    return {k if k.startswith("module.") else f"module.{k}": v for k, v in state_dict.items()}


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    candidate_dicts = []

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key, None)
            if isinstance(value, dict):
                candidate_dicts.append(value)
        if all(isinstance(k, str) for k in checkpoint.keys()):
            candidate_dicts.append(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}.")

    tried_errors = []
    for state_dict in candidate_dicts:
        for variant_name, variant in (
            ("raw", state_dict),
            ("strip_module_prefix", strip_module_prefix(state_dict)),
            ("add_module_prefix", add_module_prefix(state_dict)),
        ):
            try:
                model.load_state_dict(variant, strict=True)
                print(f"Loaded checkpoint with variant: {variant_name}")
                return
            except RuntimeError as exc:
                tried_errors.append(f"{variant_name}: {exc}")

    joined = "\n".join(tried_errors[-3:])
    raise RuntimeError(f"Could not load checkpoint: {checkpoint_path}\n{joined}")


def resolve_sample_index(dataset: VideoDataset, sample_idx: int, filename: str = None) -> int:
    if filename is not None:
        base = os.path.basename(filename)
        for idx, fn in enumerate(dataset.video_files):
            if fn == filename or fn == base:
                return idx
        raise ValueError(f"Filename not found in dataset: {filename}")

    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx must be in [0, {len(dataset)-1}], got {sample_idx}")
    return sample_idx


def save_gif(path: Path, frames, fps: int):
    imageio.mimsave(path.as_posix(), frames, fps=fps)


def save_mp4(path: Path, frames, fps: int):
    writer = imageio.get_writer(path.as_posix(), fps=fps, codec="libx264")
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    val_transform = get_val_transform()
    dataset = VideoDataset(args.video_dir, num_frames=args.num_frames, transform=val_transform)
    sample_idx = resolve_sample_index(dataset, sample_idx=args.sample_idx, filename=args.filename)

    x, y = dataset[sample_idx]          # x: (C,T,H,W)
    x_batch = x.unsqueeze(0).to(device) # (1,C,T,H,W)
    filename = dataset.video_files[sample_idx]

    model = get_model(args.model, args.pretrained, args.num_classes).to(device)
    if args.use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")

    load_state_dict_flexible(model, args.checkpoint_path, device)
    model.eval()

    target_layer = resolve_layer(model, args.target_layer)
    grad_cam = GradCAM3D(model=model, target_layer=target_layer)
    cam = grad_cam(x_batch, class_idx=args.class_idx)[0].detach().cpu()  # (T,H,W), [0,1]
    print("CAM stats:", float(cam.min()), float(cam.max()), float(cam.mean()))
    grad_cam.close()

    with torch.no_grad():
        logits = model(x_batch)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    pred_info = {"logits": logits.detach().cpu()}
    if logits.shape[1] == 1:
        pred_info["prob"] = torch.sigmoid(logits[:, 0]).detach().cpu()

    clip = x_batch[0].detach().cpu()  # (C,T,H,W), normalized
    overlay_frames = []
    for t in range(clip.shape[1]):
        frame_uint8 = denorm_frame(clip[:, t])
        cam_hw = cam[t].numpy()
        overlay_frames.append(
            overlay_frame(
                frame_uint8,
                cam_hw,
                alpha=args.alpha,
                colormap=args.colormap,
                cam_threshold=args.cam_threshold,
                cam_gamma=args.cam_gamma,
            )
        )

    stem = Path(filename).stem
    outputs = []
    if args.save_format in ("gif", "both"):
        gif_path = output_dir / f"{stem}_gradcam.gif"
        save_gif(gif_path, overlay_frames, fps=args.fps)
        outputs.append(gif_path.as_posix())
    if args.save_format in ("mp4", "both"):
        mp4_path = output_dir / f"{stem}_gradcam.mp4"
        save_mp4(mp4_path, overlay_frames, fps=args.fps)
        outputs.append(mp4_path.as_posix())

    cam_path = output_dir / f"{stem}_cam.pt"
    torch.save(
        {
            "cam": cam,  # (T,H,W), float32 in [0,1]
            "video_filename": filename,
            "sample_index": int(sample_idx),
            "label": int(y),
            "checkpoint_path": args.checkpoint_path,
            "target_layer": args.target_layer if args.target_layer else "layer4[-1].conv2",
            "class_idx": args.class_idx,
            **pred_info,
        },
        cam_path.as_posix(),
    )

    print("Saved CAM tensor:", cam_path.as_posix())
    for path in outputs:
        print("Saved overlay:", path)


if __name__ == "__main__":
    main()
