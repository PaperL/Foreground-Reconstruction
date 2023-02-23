from functools import partial

import torch
from torch import nn as nn
from LUT.models_x import *

from termcolor import colored

class CLUT(nn.Module):
    def __init__(self,
            device,
            lut
    ):
        super(CLUT, self).__init__()
        self.device = device
        self.lut = torch.from_numpy(lut).to(device)
        self.clut = torch.zeros(*(self.lut[0][0].size())).to(device)

    def forward(self, t, nlc):
        # self.clut = torch.zeros(*(self.lut[0][0].size())).to(self.device)
        luts = self.lut[t].permute((1, 2, 3, 4, 0))
        self.clut = (nlc * luts).sum(dim=4)
        # for i in range(nlc.size(0)):
        #     self.clut = self.clut + nlc[i] * self.lut[t][i]
        return self.clut


class LUTFR(nn.Module):
    def __init__(
            self,
            device,
            lut,
            lutN
    ):
        super(LUTFR, self).__init__()
        self.lutN = lutN
        # Linear Coefficient
        self.lc0 = nn.Parameter(torch.empty(lutN))
        self.lc1 = nn.Parameter(torch.empty(lutN))
        nn.init.normal_(self.lc0)
        nn.init.normal_(self.lc1)
        # Combined LUT
        self.clut = CLUT(device, lut)

        self.trilinear = TrilinearInterpolation()


    def apply_lut(self, LUT, img):
        dat = img.unsqueeze(0)
        # print(dat.size())
        _, result = self.trilinear(LUT, dat)
        return result.squeeze()


    def forward(self, gt):
        I_r = gt
        # Normalized Linear Coefficient
        nlc0 = nn.functional.softmax(self.lc0.clone(), dim=0)
        # Combined LUT
        clut0 = self.clut(0, nlc0)
        I_s = self.apply_lut(clut0, I_r)

        nlc1 = nn.functional.softmax(self.lc1.clone(), dim=0)
        clut1 = self.clut(1, nlc1)
        I_f = self.apply_lut(clut1, I_s)

        return I_f
