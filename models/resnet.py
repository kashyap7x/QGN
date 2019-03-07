import os
import sys
import torch
import torch.nn as nn
import math
import sparseconvnet as scn
from lib.nn import SynchronizedBatchNorm2d

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
                     

def conv3x3_sparse(in_planes, out_planes, stride=1):
    "3x3 sparse convolution"
    if stride == 1:
        return scn.SubmanifoldConvolution(2, in_planes, out_planes, 3, False)
    else:
        return scn.Convolution(2, in_planes, out_planes, 3, stride, False)
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = SynchronizedBatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
        
        
class TransBasicBlockSparse(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlockSparse, self).__init__()
        self.conv1 = conv3x3_sparse(inplanes, inplanes)
        # self.bn1 = scn.BatchNormalization(inplanes)
        self.relu = scn.ReLU()
        if upsample is not None and stride != 1:
            self.conv2 = scn.Deconvolution(2, inplanes, planes, 3, stride, False)
        else:
            self.conv2 = conv3x3_sparse(inplanes, planes, stride)
        # self.bn2 = scn.BatchNormalization(planes)
        self.add = scn.AddTable()
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)
        
        out = self.add([out,residual])
        out = self.relu(out)

        return out
        

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
        
class TransBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes * 4, inplanes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(inplanes)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, inplanes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.bn2 = SynchronizedBatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
        
        
class TransBottleneckSparse(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBottleneckSparse, self).__init__()
        self.conv1 = scn.SubmanifoldConvolution(2, inplanes * 4, inplanes, 1, False)
        # self.bn1 = scn.BatchNormalization(inplanes)
        if upsample is not None and stride != 1:
            self.conv2 = scn.Deconvolution(2, inplanes, planes, 3, stride, False)
        else:
            self.conv2 = conv3x3_sparse(inplanes, inplanes, stride)
        # self.bn2 = scn.BatchNormalization(inplanes)
        self.conv3 = scn.SubmanifoldConvolution(2, inplanes, planes, 1, False)
        # self.bn3 = scn.BatchNormalization(planes)
        self.relu = scn.ReLU()
        self.add = scn.AddTable()
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out = self.add([out,residual])
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
        
class ResNetTranspose(nn.Module):

    def __init__(self, transblock, layers, num_classes=150):
        self.inplanes = 512
        super(ResNetTranspose, self).__init__()
        
        self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
        self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
        self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
        self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
        
        self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
        self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
        self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
        self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
        self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
        
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64 * transblock.expansion, 3)
        
        self.final_deconv = nn.ConvTranspose2d(self.inplanes * transblock.expansion, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)
        
        self.out6_conv = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)
        self.out5_conv = nn.Conv2d(256 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_transpose(self, transblock, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                SynchronizedBatchNorm2d(planes),
            )
        elif self.inplanes * transblock.expansion != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes * transblock.expansion, planes,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes),
            )

        layers = []
        
        for i in range(1, blocks):
            layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))

        layers.append(transblock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes // transblock.expansion

        return nn.Sequential(*layers)
        
    def _make_skip_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            SynchronizedBatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x, labels=None):
        [in0, in1, in2, in3, in4] = x
        if labels:
            [lab0, lab1, lab2, lab3, lab4] = labels
        
        out6 = self.out6_conv(in4)
        
        if labels:
            mask4 = (lab4==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
        else:
            mask4 = (torch.argmax(out6, dim=1)==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
        in4 = in4 * mask4

        skip4 = self.skip4(in4)
        # upsample 1
        x = self.deconv1(skip4)
        out5 = self.out5_conv(x)
        
        if labels:
            mask3 = (lab3==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
        else:
            mask3 = (torch.argmax(out5, dim=1)==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
        in3 = in3 * mask3

        x = x + self.skip3(in3)
        # upsample 2
        x = self.deconv2(x)
        out4 = self.out4_conv(x)
        
        if labels:
            mask2 = (lab2==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
        else:
            mask2 = (torch.argmax(out4, dim=1)==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
        in2 = in2 * mask2

        x = x + self.skip2(in2)
        # upsample 3
        x = self.deconv3(x)
        out3 = self.out3_conv(x)
        
        if labels:
            mask1 = (lab1==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
        else:
            mask1 = (torch.argmax(out3, dim=1)==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
        in1 = in1 * mask1

        x = x + self.skip1(in1)
        # upsample 4
        x = self.deconv4(x)
        out2 = self.out2_conv(x)
        
        if labels:
            mask0 = (lab0==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
        else:
            mask0 = (torch.argmax(out2, dim=1)==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
        in0 = in0 * mask0

        x = x + self.skip0(in0)
        # final
        x = self.final_conv(x)
        out1 = self.final_deconv(x)

        return [out6, out5, out4, out3, out2, out1]
        
        
class ResNetTransposeSparse(nn.Module):

    def __init__(self, transblock, layers, num_classes=150):
        self.inplanes = 512
        super(ResNetTransposeSparse, self).__init__()
        
        self.dense_to_sparse = scn.DenseToSparse(2)
        self.add = scn.AddTable()
        
        self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
        self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
        self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
        self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
        
        self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
        self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
        self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
        self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
        self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
        
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64 * transblock.expansion, 3)
        
        self.final_deconv = scn.Deconvolution(2, self.inplanes * transblock.expansion, num_classes, 2, 2, True)
        
        self.out6_conv = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)
        self.out5_conv = scn.SubmanifoldConvolution(2, 256 * transblock.expansion, num_classes, 1, True)
        self.out4_conv = scn.SubmanifoldConvolution(2, 128 * transblock.expansion, num_classes, 1, True)
        self.out3_conv = scn.SubmanifoldConvolution(2, 64 * transblock.expansion, num_classes, 1, True)
        self.out2_conv = scn.SubmanifoldConvolution(2, 64 * transblock.expansion, num_classes, 1, True)
        
        self.sparse_to_dense = scn.SparseToDense(2, num_classes)
        
    def _make_transpose(self, transblock, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = scn.Sequential(
                scn.Deconvolution(2, self.inplanes * transblock.expansion, planes, 2, stride, False),
                # scn.BatchNormalization(planes),
            )
        elif self.inplanes * transblock.expansion != planes:
            upsample = scn.Sequential(
                scn.Convolution(2, self.inplanes * transblock.expansion, planes, 1, stride, False),
                # scn.BatchNormalization(planes),
            )

        layers = []
        
        for i in range(1, blocks):
            layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))

        layers.append(transblock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes // transblock.expansion

        return scn.Sequential(*layers)
        
    def _make_skip_layer(self, inplanes, planes):

        layers = scn.Sequential(
            scn.Convolution(2, inplanes, planes, 1, 1, False),
            # scn.BatchNormReLU(planes),
            scn.ReLU()
        )
        return layers

    def forward(self, x, labels=None):
        [in0, in1, in2, in3, in4] = x
        if labels:
            [lab0, lab1, lab2, lab3, lab4] = labels
        
        out6 = self.out6_conv(in4)
        
        if labels:
            mask4 = (lab4==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
        else:
            mask4 = (torch.argmax(out6, dim=1)==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
        in4 = in4 * mask4
        in4 = self.dense_to_sparse(in4)
        skip4 = self.skip4(in4)
        # upsample 1

        x = self.deconv1(skip4)
        out5 = self.sparse_to_dense(self.out5_conv(x))

        if labels:
            mask3 = (lab3==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
        else:
            mask3 = (torch.argmax(out5, dim=1)==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
        in3 = in3 * mask3
        in3 = self.dense_to_sparse(in3)
        x = self.add([x,self.skip3(in3)])
        # upsample 2
        x = self.deconv2(x)
        out4 = self.sparse_to_dense(self.out4_conv(x))
        
        if labels:
            mask2 = (lab2==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
        else:
            mask2 = (torch.argmax(out4, dim=1)==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
        in2 = in2 * mask2
        in2 = self.dense_to_sparse(in2)
        x = self.add([x,self.skip2(in2)])
        # upsample 3
        x = self.deconv3(x)
        out3 = self.sparse_to_dense(self.out3_conv(x))
        
        if labels:
            mask1 = (lab1==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
        else:
            mask1 = (torch.argmax(out3, dim=1)==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
        in1 = in1 * mask1
        in1 = self.dense_to_sparse(in1)
        x = self.add([x,self.skip1(in1)])
        # upsample 4
        x = self.deconv4(x)
        out2 = self.sparse_to_dense(self.out2_conv(x))
        
        if labels:
            mask0 = (lab0==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
        else:
            mask0 = (torch.argmax(out2, dim=1)==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
        in0 = in0 * mask0
        in0 = self.dense_to_sparse(in0)
        x = self.add([x,self.skip0(in0)])
        # final
        x = self.final_conv(x)
        out1 = self.sparse_to_dense(self.final_deconv(x))

        return [out6, out5, out4, out3, out2, out1]


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def resnet34_transpose(**kwargs):
    """Constructs a ResNet-34 transpose model.
    """
    model = ResNetTranspose(TransBasicBlock, [6, 4, 3, 3], **kwargs)
    return model
    
    
def resnet50_transpose(**kwargs):
    """Constructs a ResNet-50 transpose model.
    """
    model = ResNetTranspose(TransBottleneck, [6, 4, 3, 3], **kwargs)
    return model
    
    
def resnet101_transpose(**kwargs):
    """Constructs a ResNet-101 transpose model.
    """
    model = ResNetTranspose(TransBottleneck, [23, 4, 3, 3], **kwargs)
    return model
    
    
def resnet34_transpose_sparse(**kwargs):
    """Constructs a ResNet-34 transpose model.
    """
    model = ResNetTransposeSparse(TransBasicBlockSparse, [6, 4, 3, 3], **kwargs)
    return model
    
    
def resnet50_transpose_sparse(**kwargs):
    """Constructs a ResNet-50 transpose model.
    """
    model = ResNetTransposeSparse(TransBottleneckSparse, [6, 4, 3, 3], **kwargs)
    return model
    
    
def resnet101_transpose_sparse(**kwargs):
    """Constructs a ResNet-101 transpose model.
    """
    model = ResNetTransposeSparse(TransBottleneckSparse, [23, 4, 3, 3], **kwargs)
    return model
    

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
