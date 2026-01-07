# Miguel Hernández University of Elche
# Institute for Engineering Research of Elche (I3E)
# Automation, Robotics and Computer Vision lab (ARCV)
# Author: Juan José Cabrera Mora

import torch
from config import PARAMS
import torch.nn as nn
import MinkowskiEngine as ME
from model.residual_blocks import *
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor
# from resnet import ResNetBase

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, batch: ME.SparseTensor):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x).F


class ResNet(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class MinkUNeXt(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    BLOCK = MinkNeXtBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1
    POOLING = None
    FUSION = None
    FINAL_LAYER = 'conv1x1x1'
    add_GFLM = False
    add_1x1x1_convs = False
    out_dimension = 512

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3, agg_fusion=True):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)
        #self.attentions = nn.ModuleList()   # Bottom-up blocks

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)

        if PARAMS.use_cross_entropy:
            self.classification_head = ClassificationHead()

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.GeM_pool = GeM(input_dim=out_channels)

    def forward(self, batch):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        descriptor = self.GeM_pool(out)
        if PARAMS.use_cross_entropy:
            logits = self.classification_head(out.detach())
            return {'global': descriptor, 'logits': logits}
        else:
            return {'global': descriptor}

class ClassificationHead(nn.Module):
    def __init__(self, in_features=192, hidden_dim=None, n_classes=3, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_features // 2, 32)
        self.mlp = nn.Sequential(
            ME.MinkowskiLinear(in_features, hidden_dim),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(dropout),
            ME.MinkowskiLinear(hidden_dim, n_classes)
        )
    def forward(self, x):
        out = self.mlp(x)
        logits = out.F        # (N_points, C)
        batch_idxs = out.C[:, 0].long()  # (N_points,)

        # pooling por scan
        B = batch_idxs.max().item() + 1
        C = logits.shape[1]
        device = logits.device

        pooled = torch.zeros(B, C, device=device)
        counts = torch.zeros(B, device=device)

        pooled.scatter_add_(0,
            batch_idxs.unsqueeze(1).expand(-1, C),
            logits
        )
        counts.scatter_add_(0,
            batch_idxs,
            torch.ones_like(batch_idxs, dtype=torch.float, device=device)
        )

        pooled = pooled / counts.unsqueeze(1)

        return pooled

model = MinkUNeXt(in_channels=1, out_channels=192, D=3)

