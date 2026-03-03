import torch
import torch.nn.functional as F

class GradCAM3D:
    """
    3D Grad-CAM for video models.
    Returns CAM: (N, T, H, W) normalized to [0,1] per sample.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None  # (N, C, t, h, w)
        self.gradients = None    # (N, C, t, h, w)
        self.handles = []
        self._register()

    def _register(self):
        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @staticmethod
    def _norm(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # cam: (N, T, H, W)
        n = cam.shape[0]
        flat = cam.view(n, -1)
        mn = flat.min(dim=1).values.view(n, 1, 1, 1)
        mx = flat.max(dim=1).values.view(n, 1, 1, 1)
        return (cam - mn) / (mx - mn + eps)

    def __call__(self, x: torch.Tensor, class_idx=None) -> torch.Tensor:
        """
        x: (N, C, T, H, W) normalized clip tensor.
        class_idx:
          - None: for binary logits (B,1) uses that score; for multiclass uses argmax per sample
          - int: class index to explain
          - Tensor/list shape (N,): per-sample class indices
        """
        self.model.eval()

        # IMPORTANT: do NOT wrap in torch.no_grad()
        x = x.requires_grad_(True)

        logits = self.model(x)  # (N,1) in your code, or (N,K) in multiclass
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        N, K = logits.shape

        # choose target score
        if K == 1:
            score = logits[:, 0].sum()
        else:
            if class_idx is None:
                idx = logits.argmax(dim=1)
            elif isinstance(class_idx, int):
                idx = torch.full((N,), class_idx, device=logits.device, dtype=torch.long)
            else:
                idx = torch.as_tensor(class_idx, device=logits.device, dtype=torch.long)

            score = logits[torch.arange(N, device=logits.device), idx].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations/gradients captured. Check target_layer selection.")

        acts = self.activations          # (N, C, t, h, w)
        grads = self.gradients           # (N, C, t, h, w)

        # channel weights = GAP over (t,h,w)
        w = grads.mean(dim=(2,3,4), keepdim=True)  # (N,C,1,1,1)

        cam = (w * acts).sum(dim=1)  # (N,t,h,w)
        cam = F.relu(cam)

        # upsample to input (T,H,W)
        cam = cam.unsqueeze(1)  # (N,1,t,h,w)
        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3], x.shape[4]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)  # (N,T,H,W)

        cam = self._norm(cam.detach())
        return cam