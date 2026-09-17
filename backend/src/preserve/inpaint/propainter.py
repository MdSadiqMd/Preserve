"""ProPainter backend for propagation-first video inpainting.

ProPainter (ICCV 2023) uses flow completion + image/feature propagation
before sparse transformer, making it ideal for cases where the hidden
background appears elsewhere in the shot.

Reference: https://github.com/sczhou/ProPainter
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from preserve.config import settings
from preserve.inpaint.base import InpaintBackend, InpaintRequest, InpaintResult


class ProPainterBackend(InpaintBackend):
    """ProPainter video inpainting backend with MPS support."""

    def __init__(self, model_dir: Path | None = None, config: dict[str, Any] | None = None):
        self._config = config or settings.get_backend_config("propainter")
        self.model_dir = model_dir or settings.model_dir / "propainter"
        self._inpaint_model = None
        self._flow_model = None
        self._raft = None
        self._device = None

    @property
    def name(self) -> str:
        return self._config.get("name", "ProPainter")

    @property
    def supports_mps(self) -> bool:
        return self._config.get("supports_mps", True)

    def is_available(self) -> bool:
        return self._inpaint_model is not None

    def load(self) -> None:
        """Load ProPainter models."""
        from preserve.propainter_model.modules.flow_comp_raft import RAFT_bi
        from preserve.propainter_model.propainter import InpaintGenerator
        from preserve.propainter_model.recurrent_flow_completion import RecurrentFlowCompleteNet

        self._device = settings.get_device()
        checkpoints = self._config.get("checkpoints", {})

        inpaint_path = self.model_dir / checkpoints.get("model", "ProPainter.pth")
        flow_path = self.model_dir / checkpoints.get("flow", "recurrent_flow_completion.pth")
        raft_path = self.model_dir / checkpoints.get("raft", "raft-things.pth")

        if not inpaint_path.exists():
            raise FileNotFoundError(f"ProPainter checkpoint not found: {inpaint_path}")
        if not flow_path.exists():
            raise FileNotFoundError(f"Flow completion checkpoint not found: {flow_path}")
        if not raft_path.exists():
            raise FileNotFoundError(f"RAFT checkpoint not found: {raft_path}")

        self._raft = RAFT_bi(str(raft_path), device=self._device)

        self._flow_model = RecurrentFlowCompleteNet(str(flow_path))
        for p in self._flow_model.parameters():
            p.requires_grad = False
        self._flow_model.to(self._device)
        self._flow_model.eval()

        self._inpaint_model = InpaintGenerator(model_path=str(inpaint_path))
        self._inpaint_model.to(self._device)
        self._inpaint_model.eval()

    def unload(self) -> None:
        self._inpaint_model = None
        self._flow_model = None
        self._raft = None
        if self._device:
            if self._device.type == "mps":
                torch.mps.empty_cache()
            elif self._device.type == "cuda":
                torch.cuda.empty_cache()
        self._device = None

    def _to_tensor(self, frames: list[NDArray[np.uint8]]) -> torch.Tensor:
        """Convert frames to tensor [1, T, C, H, W] in [-1, 1]."""
        tensor = np.stack(frames, axis=0)
        tensor = torch.from_numpy(tensor).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.unsqueeze(0)
        tensor = tensor * 2 - 1
        return tensor.to(self._device)

    def _mask_to_tensor(self, masks: list[NDArray[np.uint8]]) -> torch.Tensor:
        """Convert masks to tensor [1, T, 1, H, W] in [0, 1]."""
        tensor = np.stack(masks, axis=0)
        tensor = torch.from_numpy(tensor).float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self._device)

    def _free_cache(self) -> None:
        if self._device is None:
            return
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _complete_flow_chunked(
        self,
        gt_flows_bi: tuple[torch.Tensor, torch.Tensor],
        flow_masks_t: torch.Tensor,
        subvideo_length: int,
        pad_len: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run flow completion in overlapping windows.

        Completing the whole clip at once allocates a tensor proportional to
        T H W, which exhausts unified memory at higher resolutions. Windows
        overlap by pad_len frames and the padding is trimmed before
        concatenation so the seams stay consistent.
        """
        flow_length = gt_flows_bi[0].size(1)
        if flow_length <= subvideo_length:
            pred = self._flow_model.forward_bidirect_flow(gt_flows_bi, flow_masks_t)[0]
            return self._flow_model.combine_flow(gt_flows_bi, pred, flow_masks_t)

        forward_chunks, backward_chunks = [], []
        for start in range(0, flow_length, subvideo_length):
            s_f = max(0, start - pad_len)
            e_f = min(flow_length, start + subvideo_length + pad_len)
            trim_start = start - s_f
            trim_end = e_f - min(flow_length, start + subvideo_length)

            window = (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f])
            mask_window = flow_masks_t[:, s_f : e_f + 1]

            pred = self._flow_model.forward_bidirect_flow(window, mask_window)[0]
            pred = self._flow_model.combine_flow(window, pred, mask_window)

            keep_end = e_f - s_f - trim_end
            forward_chunks.append(pred[0][:, trim_start:keep_end])
            backward_chunks.append(pred[1][:, trim_start:keep_end])
            self._free_cache()

        return torch.cat(forward_chunks, dim=1), torch.cat(backward_chunks, dim=1)

    def _propagate_chunked(
        self,
        frames_t: torch.Tensor,
        masked_frames: torch.Tensor,
        masks_dilated_t: torch.Tensor,
        pred_flows_bi: tuple[torch.Tensor, torch.Tensor],
        h_new: int,
        w_new: int,
        subvideo_length: int,
        pad_len: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run image propagation in overlapping windows, as above."""
        video_length = frames_t.size(1)
        window_len = min(100, subvideo_length)

        if video_length <= window_len:
            b, t = masks_dilated_t.size(0), masks_dilated_t.size(1)
            prop_imgs, local_masks = self._inpaint_model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_t, "nearest"
            )
            updated = (
                frames_t * (1 - masks_dilated_t)
                + prop_imgs.view(b, t, 3, h_new, w_new) * masks_dilated_t
            )
            return updated, local_masks.view(b, t, 1, h_new, w_new)

        frame_chunks, mask_chunks = [], []
        for start in range(0, video_length, window_len):
            s_f = max(0, start - pad_len)
            e_f = min(video_length, start + window_len + pad_len)
            trim_start = start - s_f
            trim_end = e_f - min(video_length, start + window_len)

            b, t = masks_dilated_t[:, s_f:e_f].size(0), masks_dilated_t[:, s_f:e_f].size(1)
            flows_window = (
                pred_flows_bi[0][:, s_f : e_f - 1],
                pred_flows_bi[1][:, s_f : e_f - 1],
            )

            prop_imgs, local_masks = self._inpaint_model.img_propagation(
                masked_frames[:, s_f:e_f],
                flows_window,
                masks_dilated_t[:, s_f:e_f],
                "nearest",
            )
            updated = (
                frames_t[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f])
                + prop_imgs.view(b, t, 3, h_new, w_new) * masks_dilated_t[:, s_f:e_f]
            )

            keep_end = e_f - s_f - trim_end
            frame_chunks.append(updated[:, trim_start:keep_end])
            mask_chunks.append(local_masks.view(b, t, 1, h_new, w_new)[:, trim_start:keep_end])
            self._free_cache()

        return torch.cat(frame_chunks, dim=1), torch.cat(mask_chunks, dim=1)

    def _get_ref_index(
        self,
        mid_neighbor_id: int,
        neighbor_ids: list[int],
        length: int,
        ref_stride: int,
        ref_num: int,
    ) -> list[int]:
        """Get reference frame indices for transformer."""
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    @torch.inference_mode()
    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        """Run ProPainter inference."""
        if not self.is_available():
            raise RuntimeError("ProPainter not loaded. Call load() first.")

        model_settings = self._config.get("settings", {})
        neighbor_length = model_settings.get("neighbor_length", 10)
        ref_stride = model_settings.get("ref_stride", 10)
        subvideo_length = model_settings.get("subvideo_length", 80)
        raft_iters = model_settings.get("raft_iters", 20)
        flow_mask_dilates = model_settings.get("flow_mask_dilates", 8)
        mask_dilates = model_settings.get("mask_dilates", 5)

        frames = request.frames
        masks_input = request.masks

        h, w = frames[0].shape[:2]
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        h_new = h + h_pad
        w_new = w + w_pad

        if h_pad > 0 or w_pad > 0:
            frames = [
                cv2.copyMakeBorder(f, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE) for f in frames
            ]
            masks_input = np.pad(masks_input, ((0, 0), (0, h_pad), (0, w_pad)), mode="edge")

        flow_masks = []
        masks_dilated = []
        for m in masks_input:
            m_bin = (m > 127).astype(np.uint8)
            flow_m = (
                scipy.ndimage.binary_dilation(m_bin, iterations=flow_mask_dilates).astype(np.uint8)
                * 255
            )
            flow_masks.append(flow_m)

            dilated_m = (
                scipy.ndimage.binary_dilation(m_bin, iterations=mask_dilates).astype(np.uint8) * 255
            )
            masks_dilated.append(dilated_m)

        frames_t = self._to_tensor(frames)
        flow_masks_t = self._mask_to_tensor(flow_masks)
        masks_dilated_t = self._mask_to_tensor(masks_dilated)

        video_length = frames_t.size(1)

        short_clip_len = (
            12 if w_new <= 640 else (8 if w_new <= 720 else (4 if w_new <= 1280 else 2))
        )

        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f_idx in range(0, video_length, short_clip_len):
                end_f = min(video_length, f_idx + short_clip_len)
                if f_idx == 0:
                    flows_f, flows_b = self._raft(frames_t[:, f_idx:end_f], iters=raft_iters)
                else:
                    flows_f, flows_b = self._raft(frames_t[:, f_idx - 1 : end_f], iters=raft_iters)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self._raft(frames_t, iters=raft_iters)

        pred_flows_bi = self._complete_flow_chunked(gt_flows_bi, flow_masks_t, subvideo_length)

        masked_frames = frames_t * (1 - masks_dilated_t)
        updated_frames, updated_masks = self._propagate_chunked(
            frames_t,
            masked_frames,
            masks_dilated_t,
            pred_flows_bi,
            h_new,
            w_new,
            subvideo_length,
        )

        ori_frames = frames
        comp_frames = [None] * video_length

        neighbor_stride = neighbor_length // 2

        ref_num = subvideo_length // ref_stride if video_length > subvideo_length else -1

        for f_idx in tqdm(range(0, video_length, neighbor_stride), desc="Inpainting"):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f_idx - neighbor_stride),
                    min(video_length, f_idx + neighbor_stride + 1),
                )
            ]
            ref_ids = self._get_ref_index(f_idx, neighbor_ids, video_length, ref_stride, ref_num)

            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )

            l_t = len(neighbor_ids)
            pred_img = self._inpaint_model(
                selected_imgs,
                selected_pred_flows_bi,
                selected_masks,
                selected_update_masks,
                l_t,
            )

            pred_img = pred_img.view(-1, 3, h_new, w_new)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

            binary_masks = (
                masks_dilated_t[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy()
            )

            for i, idx in enumerate(neighbor_ids):
                mask_np = binary_masks[i]
                img = pred_img[i].astype(np.uint8) * mask_np + ori_frames[idx] * (1 - mask_np)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    ).astype(np.uint8)

            self._free_cache()

        result_frames = []
        for f in comp_frames:
            if f is None:
                result_frames.append(frames[0][:h, :w])
            else:
                result_frames.append(f[:h, :w].astype(np.uint8))

        return InpaintResult(frames=result_frames)


def create_propainter_backend() -> ProPainterBackend:
    """Factory function for ProPainter backend."""
    return ProPainterBackend()
