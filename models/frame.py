import timm
import torch
from ptflops import get_model_complexity_info
from torch import nn

from models.normalization import L2N
from models.pooling import GeM, SpatialAttention2d, MultiAtrousModule, OrthogonalFusion


# 55.37 MMac 2.54 M, torch.size([16, 576])
class MobileNetV3Small(nn.Module):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.base = timm.create_model('mobilenetv3_small_100', pretrained=True)
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base.forward_features(x)
        x = self.pool(x)
        x = x[:, :, 0, 0]
        x = self.norm(x)
        return x


# 938.31 MMac 12.23 M, torch.Size([16, 1536])
class EfficientNetb3(nn.Module):
    def __init__(self):
        super(EfficientNetb3, self).__init__()
        self.base = timm.create_model('tf_efficientnet_b3', pretrained=True)
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base.forward_features(x)
        x = self.pool(x)
        x = x[:, :, 0, 0]
        x = self.norm(x)
        return x


# 1.11 GMac 44.46 M, torch.size([16, 384])
class HybridViT(nn.Module):
    def __init__(self):
        super(HybridViT, self).__init__()
        self.base = timm.create_model('vit_small_r26_s32_224_in21k', pretrained=True)
        self.norm = L2N()

    def forward(self, x):
        x = self.base.forward_features(x)
        x = x[:, 0]
        x = self.norm(x)
        return x

# 265.62 MMac 3.0 M, torch.Size([16, 768])
class DOLG(nn.Module):
    def __init__(self, arch, stride=(2, 2), dilations=[3, 6, 9]):
        super(DOLG, self).__init__()

        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, global_pool="", in_chans=3,
                                          features_only=True)

        if ("efficientnet" in arch) & (stride is not None):
            self.backbone.conv_stem.stride = stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']

        feature_dim_l_g = 512
        fusion_out = 2 * feature_dim_l_g

        self.global_pool = GeM()
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = 256 * 3

        self.neck = nn.Sequential(
            nn.Linear(fusion_out, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )

        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, dilations)
        self.conv_g = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g = nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()
        self.norm = L2N()

    def forward(self, x):
        x = self.backbone(x)
        x_l = x[-2]
        x_g = x[-1]

        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)

        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)

        x_g = self.global_pool(x_g)
        x_g = x_g[:, :, 0, 0]

        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:, :, 0, 0]

        x_emb = self.neck(x_fused)
        return x_emb


# 474.68 MMac 2.61 M, torch.Size([16, 768])
class DOLG_FF(nn.Module):
    def __init__(self, arch, stride=(2, 2), dilations=[3, 6, 9]):
        super(DOLG_FF, self).__init__()

        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, global_pool="", in_chans=3,
                                          features_only=True)

        if ("efficientnet" in arch) & (stride is not None):
            self.backbone.conv_stem.stride = stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']

        feature_dim_l_g = 512
        fusion_out = 2 * feature_dim_l_g

        self.global_pool = GeM()
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = 384 * 2

        self.neck = nn.Sequential(
            nn.Linear(fusion_out, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )

        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, dilations)
        self.conv_g = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g = nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()
        self.norm = L2N()

        self.global_pool2 = GeM()
        self.fusion_pool2 = nn.AdaptiveAvgPool2d(1)
        self.neck2 = nn.Sequential(
            nn.Linear(fusion_out, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )
        self.mam2 = MultiAtrousModule(backbone_out_1, feature_dim_l_g, dilations)
        self.conv_g2 = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g2 = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g2 = nn.SiLU(inplace=True)
        self.attention2d2 = SpatialAttention2d(feature_dim_l_g)
        self.fusion2 = OrthogonalFusion()

    def forward(self, x):
        x = self.backbone(x)

        x_l = x[-2]
        x_g = x[-1]

        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)

        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)

        x_g = self.global_pool(x_g)
        x_g = x_g[:, :, 0, 0]

        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:, :, 0, 0]

        x_l2 = torch.flip(x[-2], [1, 2])
        x_g2 = torch.flip(x[-1], [1, 2])

        x_l2 = self.mam2(x_l2)
        x_l2, att_score = self.attention2d2(x_l2)

        x_g2 = self.conv_g2(x_g2)
        x_g2 = self.bn_g2(x_g2)
        x_g2 = self.act_g2(x_g2)

        x_g2 = self.global_pool2(x_g2)
        x_g2 = x_g2[:, :, 0, 0]

        x_fused2 = self.fusion2(x_l2, x_g2)
        x_fused2 = self.fusion_pool2(x_fused2)
        x_fused2 = x_fused2[:, :, 0, 0]

        x_emb1 = self.neck(x_fused)
        x_emb2 = self.neck2(x_fused2)

        x_emb = torch.cat((x_emb1, x_emb2), dim=1)
        x_emb = self.norm(x_emb)

        return x_emb


if __name__ == '__main__':
    # net = DOLG_FF(arch='mobilenetv3_small_100')
    net = DOLG(arch='resnet50')

    # net = MobileNetV3Small()
    macs, params = get_model_complexity_info(net, (3, 224, 224))
    out = net(torch.randn(16, 3, 224, 224))

    print(macs, params)
    import pdb;

    pdb.set_trace()
    # print(out.shape)
