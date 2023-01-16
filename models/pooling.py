import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class MAC(nn.Module):

    def __init__(self):
        super(MAC, self).__init__()

    def mac(self, x):
        return F.max_pool2d(x, (x.size(-2), x.size(-1)))

    def forward(self, x):
        return self.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def spoc(self, x):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

    def forward(self, x):
        return self.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def rmac(self, x, L=3, eps=1e-6):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

        for l in range(1, L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                    v += vt

        return v

    def forward(self, x):
        return self.rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()

        self.d0 = nn.Conv2d(in_chans, 256, kernel_size=3, dilation=dilations[0], padding='same')
        self.d1 = nn.Conv2d(in_chans, 256, kernel_size=3, dilation=dilations[1], padding='same')
        self.d2 = nn.Conv2d(in_chans, 256, kernel_size=3, dilation=dilations[2], padding='same')
        self.conv1 = nn.Conv2d(256 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)
        self.bn = nn.BatchNorm2d(512)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score
        '''
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1)

        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)

        x = att * feature_map_norm
        return x, att_score


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):
        bs, c, w, h = fl.shape

        fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))
        fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
        fg_norm = torch.norm(fg, dim=1)

        fl_proj = (fl_dot_fg / fg_norm[:, None, None, None]) * fg[:, :, None, None]
        fl_orth = fl - fl_proj

        f_fused = torch.cat([fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
        return f_fused


if __name__ == '__main__':
    print(MAC())
    print(SPoC())
    print(GeM())
    print(RMAC())
