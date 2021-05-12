import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

# MLP block for Voxelized approach
def mlp_block(in_channel, out_channel):
    return nn.Sequential(
        ME.MinkowskiLinear(in_channel, out_channel, bias=False),
        ME.MinkowskiBatchNorm(out_channel),
        ME.MinkowskiLeakyReLU()
    )

# Convolution Block for Voxelized Approach
def convolution_block(in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
        ME.MinkowskiConvolution(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            dimension=3
        ),
        ME.MinkowskiBatchNorm(out_channel),
        ME.MinkowskiLeakyReLU()
    )

#  PointNet model to process the points into voxels
class PointNet(ME.MinkowskiNetwork):
    def __init__(self, embedding_dim = 128, dimension = 3):
        ME.MinkowskiNetwork.__init__(self, dimension)

        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(dimension, embedding_dim//4, bias=True),
            ME.MinkowskiBatchNorm(embedding_dim//4),
            ME.MinkowskiReLU()
        )

        self.linear2 = nn.Sequential(
            ME.MinkowskiLinear(embedding_dim//4, embedding_dim//2, bias=True),
            ME.MinkowskiBatchNorm(embedding_dim//2),
            ME.MinkowskiReLU()
        )

        self.linear3 = ME.MinkowskiLinear(embedding_dim//2, embedding_dim-3, bias=True)

    def forward(self, x):
        out = self.linear3(self.linear2(self.linear1(x)))
        return out

#  Final Model
class PoxelNet(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        self.network_initialization(in_channel, out_channel, embedding_channel=embedding_channel, kernel_size=3, D=D)
        self.weight_initialization()

    def network_initialization(self, in_channel, out_channel, embedding_channel, kernel_size, D=3):
        
        self.pointnet = PointNet(embedding_dim = 128)

        self.conv1 = convolution_block(128, 156, kernel_size=kernel_size, stride=1)
        self.conv2 = convolution_block(156, 192, kernel_size=kernel_size, stride=2)
        self.conv3 = convolution_block(192, 228, kernel_size=kernel_size, stride=2)
        self.conv4 = convolution_block(228, 256, kernel_size=kernel_size, stride=2)
        self.conv5 = nn.Sequential(
            convolution_block(156+192+228+256, embedding_channel // 4, kernel_size=3, stride=2),
            convolution_block(embedding_channel // 4, embedding_channel // 2, kernel_size=3, stride=2),
            convolution_block(embedding_channel // 2, embedding_channel, kernel_size=3, stride=2),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

    # Kaiming Normal Weight initialization
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):

        pointnet_output = self.pointnet(x)
        volumetric_input = pointnet_output.sparse().cat_slice(x).sparse() 

        y = self.conv1(volumetric_input)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F