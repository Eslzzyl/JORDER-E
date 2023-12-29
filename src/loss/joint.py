import torch.nn as nn
import torch.nn.functional as F


class Joint(nn.Module):
    def __init__(self):
        super(Joint, self).__init__()

    def forward(self, sr, hr, lr, detect_map):
        rain_map_gt = hr - lr
        rain_map_gt = rain_map_gt.abs() > 0.00001
        rain_map_gt = rain_map_gt.narrow(1, 0, 1).float()

        detect_map = detect_map.narrow(1, 0, 1)

        loss = F.smooth_l1_loss(sr, hr) + 10 * F.smooth_l1_loss(detect_map, rain_map_gt)

        return loss
