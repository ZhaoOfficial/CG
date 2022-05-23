from typing import Tuple

import torch
import torch.nn as nn

class RaySamplerBox(nn.Module):
    def __init__(self, num_coarse_sample: int) -> None:
        super(RaySamplerBox, self).__init__()
        self.num_coarse_sample = num_coarse_sample

    def intersect(self, rays: torch.Tensor, bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rays.size() = (N, 6)
        # bbox.size() = (N, 3, 2), minimum and maximum
        rayo = rays[:, :3]
        rayd = rays[:, 3:]

        # intersection time
        inv_rayd = 1.0 / rayd
        t_i = (bbox[..., 0] - rayo) * inv_rayd
        t_o = (bbox[..., 1] - rayo) * inv_rayd

        # compare the intersection time
        swap_mask = inv_rayd > 0.0
        # t_min, t_max.size() = (N, 6)
        t_min = torch.where(swap_mask, t_i, t_o)
        t_max = torch.where(swap_mask, t_o, t_i)

        return t_min, t_max

    def forward(self, rays: torch.Tensor, bbox: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # rays.size() = (N, 6)
        # bbox.size() = (N, 3, 2), minimum and maximum

        N = rays.size(0)
        S = self.num_coarse_sample
        rayo = rays[:, :3]
        rayd = rays[:, 3:]
        # t_min, t_max.size() = (N, 3)
        t_min, t_max = self.intersect(rays, bbox)

        # intervals.size() = (1, S)
        intervals = torch.arange(S, dtype=torch.float32, device=self.device).reshape(1, -1)
        jitter = torch.rand(N, S, device=rays.device)
        # t_sample.size() = (N, S, 3)
        t_sample = (intervals + jitter) * ((t_max - t_min) / S + t_min).unsqueeze(1)

        # point_sample.size() = (N, S, 3)
        point_sample = rayo.reshape(N, 1, 3) + rayd.reshape(N, 1, 3) * t_sample
        # mask.size() = (N, S, 3)
        mask = torch.nan_to_num(t_max - t_min, nan=0.0, posinf=0.0, neginf=0.0)
        mask = mask > 1e-5

        return t_sample, point_sample, mask

class RaySamplerNearFar(nn.Module):
    def __init__(self, num_coarse_sample) -> None:
        super(RaySamplerNearFar, self).__init__()

        self.num_coarse_sample = num_coarse_sample

    def forward(self, rays: torch.Tensor, near_far: torch.Tensor) -> torch.Tensor:
        # rays.size() = (N, 6)
        # near_far.size() = (N, 2)

        N = rays.size(0)
        S = self.num_coarse_sample

        rayo = rays[:, :3]
        rayd = rays[:, 3:]

        # interval.size() = (N, S + 1)
        interval = torch.linspace(0.0, 1.0, S + 1, device=rays.device).expand(N, S)
        # depth.size() = (N, S + 1)
        # endpoints of each bin
        diff = (near_far[:, 1:2] - near_far[:, 0:1]).expand(-1, S + 1)
        depth = near_far[:, 0:1].expand(-1, S + 1) + diff * interval

        #* Perturb sampling time along each ray
        perturb = torch.rand(N, S)
        # t_sample.size() = (N, S, 1)
        t_sample = (depth[:, :S] + (depth[:, 1:] - depth[:, :S]) * perturb).unsqueeze(1)

        # t_sample.size() = (N, S, 3)
        point_sample = rayo.reshape(N, 1, 3) + rayd.reshape(N, 1, 3) * t_sample

        return point_sample
