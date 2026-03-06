import torch
import torch.nn.functional as F


class GradCAM3D:
    """
    3D Grad-CAM for video models.

    Input:
      - x: Tensor of shape (N, C, T, H, W), normalized input clip.

    Output:
      - cam: Tensor of shape (N, T, H, W), normalized to [0, 1] per sample
             and upsampled to input (T, H, W) with trilinear interpolation.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else self._default_target_layer()
        self.activations = None  # (N, C, t, h, w)
        self.gradients = None    # (N, C, t, h, w)
        self.handles = []
        self._register_hooks()

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _default_target_layer(self) -> torch.nn.Module:
        """
        Default for torchvision r3d_18:
          model.layer4[-1].conv2
        """
        base_model = self._unwrap_model()
        try:
            return base_model.layer4[-1].conv2
        except Exception as exc:
            raise ValueError(
                "Could not resolve default target layer. "
                "Pass `target_layer` explicitly."
            ) from exc

    def _register_hooks(self):
        def forward_hook(_, __, output):
            # Capture forward activations and attach a tensor-level gradient hook.
            # This avoids module backward-hook + in-place op conflicts in some torchvision backbones.
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "Target layer output must be a Tensor for Grad-CAM. "
                    f"Got {type(output)}."
                )
            self.activations = output

            def _save_grad(grad):
                self.gradients = grad

            output.register_hook(_save_grad)

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __del__(self):
        # Best-effort hook cleanup if caller forgets to call close()
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalize_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # cam: (N, T, H, W)
        n = cam.shape[0]
        flat = cam.reshape(n, -1)
        min_v = flat.min(dim=1).values.reshape(n, 1, 1, 1)
        max_v = flat.max(dim=1).values.reshape(n, 1, 1, 1)
        return (cam - min_v) / (max_v - min_v + eps)

    @staticmethod
    def _resolve_class_indices(logits: torch.Tensor, class_idx):
        n, k = logits.shape
        if class_idx is None:
            return logits.argmax(dim=1)
        if isinstance(class_idx, int):
            idx = torch.full((n,), class_idx, device=logits.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(class_idx, device=logits.device, dtype=torch.long)
            if idx.ndim != 1 or idx.numel() != n:
                raise ValueError(
                    f"class_idx must be int or shape (N,), got shape {tuple(idx.shape)} for N={n}."
                )

        if idx.min().item() < 0 or idx.max().item() >= k:
            raise ValueError(f"class_idx values must be in [0, {k - 1}].")
        return idx

    def __call__(self, x: torch.Tensor, class_idx=None) -> torch.Tensor:
        """
        Args:
          x: (N, C, T, H, W)
          class_idx:
            - None:
                - binary head (N,1): explain logit[:,0]
                - multiclass (N,K): explain argmax class per sample
            - int: explain one class for all samples (multiclass)
            - list/Tensor shape (N,): class index per sample (multiclass)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (N,C,T,H,W), got {tuple(x.shape)}.")

        self.model.eval()
        self.activations = None
        self.gradients = None

        # Do not wrap in torch.no_grad(), we need gradients for CAM.
        x = x.requires_grad_(True)
        logits = self.model(x)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.ndim != 2:
            raise ValueError(f"Expected model output shape (N,1) or (N,K), got {tuple(logits.shape)}.")

        n, k = logits.shape
        if k == 1:
            # For binary head: class_idx=1 -> seizure evidence (logit)
            #                  class_idx=0 -> non-seizure evidence (-logit)
            if class_idx is None or class_idx == 1:
                score = logits[:, 0].sum()
            elif class_idx == 0:
                score = (-logits[:, 0]).sum()
            else:
                raise ValueError("For binary head (N,1), class_idx must be 0 (nonseizure) or 1 (seizure).")
        else:
            idx = self._resolve_class_indices(logits, class_idx)
            score = logits[torch.arange(n, device=logits.device), idx].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations/gradients captured. Check target_layer selection.")

        acts = self.activations
        grads = self.gradients

        # Channel weights: global-average pool gradients over (t, h, w)
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # (N,C,1,1,1)
        cam = F.relu((weights * acts).sum(dim=1))          # (N,t,h,w)

        # Upsample to input resolution (T,H,W)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3], x.shape[4]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)

        cam = self._normalize_cam(cam.detach())
        return cam
