from functools import partial

import torch
from torch import nn as nn
from LUT.models_x import *

class LUTFR(nn.Module):
    def __init__(
            self,
            device,
            lut
    ):
        super(LUTFR, self).__init__()
        self.device = device
        lut = torch.from_numpy(lut)
        self.lut = lut
        self.lut.to(device)

        self.L0 = torch.nn.Parameter(torch.FloatTensor(350))
        self.CL0 = torch.zeros(*(self.lut[0][0].size()))
        self.L1 = torch.nn.Parameter(torch.FloatTensor(350))
        self.CL1 = torch.zeros(*(self.lut[1][0].size()))
        self.L0.to(device)
        self.CL0.to(device)
        self.L1.to(device)
        self.CL1.to(device)
        
    def apply_lut(LUT, img):
        dat = img.unsqueeze(0)
        # print(dat.size())
        trilinear = TrilinearInterpolation()
        _, result = trilinear(LUT, dat)
        return result.squeeze()

    def forward(self, gt):
        I_r = gt
        nl0 = nn.functional.softmax(self.L0, dim=0)
        self.CL0 = torch.zeros(*(self.lut[0][0].size()))
        for i in range(nl0.size(0)):
            self.CL0 += self.lut[0][i]
        I_s = LUTFR.apply_lut(self.CL0, I_r)

        nl1 = nn.functional.softmax(self.L1, dim=0)
        self.CL1 = torch.zeros(*(self.lut[1][0].size()))
        for i in range(nl1.size(0)):
            self.CL1 += self.lut[1][i]
        I_f = LUTFR.apply_lut(self.CL1, I_s)
        return I_f
