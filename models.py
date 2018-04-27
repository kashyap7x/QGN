import torch
import torch.nn as nn
import torchvision
import resnet
from utils import gather

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'vgg16_dilated8':
            orig_vgg = torchvision.models.vgg16(pretrained=pretrained)
            conv5 = (24, 26, 28)
            pool4n5 = (23, 30)
            net_encoder = VggDilated(orig_vgg,
                                     conv5,
                                     pool4n5,
                                     dropout2d=True)
        elif arch == 'vgg19_dilated8':
            orig_vgg = torchvision.models.vgg19(pretrained=pretrained)
            conv5 = (28, 30, 32, 34)
            pool4n5 = (27, 36)
            net_encoder = VggDilated(orig_vgg,
                                     conv5,
                                     pool4n5,
                                     dropout2d=True)
        elif arch == 'resnet34':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet34_quad':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetQuad(orig_resnet)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50_quad':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetQuad(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_encoder

    def build_decoder(self, arch='c1_bilinear', fc_dim=2048, num_class=150,
                      segSize=384, weights='', use_softmax=False):
        if arch == 'c1_bilinear':
            net_decoder = C1Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        elif arch == 'c5_bilinear':
            net_decoder = C5Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        elif arch == 'psp_bilinear':
            net_decoder = PSPBilinear(num_class=num_class,
                                      fc_dim=fc_dim,
                                      segSize=segSize,
                                      use_softmax=use_softmax)
        elif arch == 'quad_bilinear':
            net_decoder = QuadBilinear(num_class=num_class,
                                      segSize=segSize,
                                      use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_decoder


class VggDilated(nn.Module):
    def __init__(self, orig_vgg, conv5, pool4n5, dropout2d=True):
        super(VggDilated, self).__init__()

        # make conv5 dilated
        for i in conv5:
            orig_vgg.features[i].dilation = (2, 2)
            orig_vgg.features[i].padding = (2, 2)
        # take away pool4 and pool5
        modules = [x for i, x in enumerate(orig_vgg.features)
                   if i not in pool4n5]
        self.features = nn.Sequential(*(modules))

        # convert fc weights into conv1x1 weights
        self.conv6 = nn.Conv2d(512, 4096, 7, 1, 12, 4)
        self.conv6.weight.data.copy_(
            orig_vgg.classifier[0].weight.data.resize_(4096, 512, 7, 7))
        self.conv6.bias.data.copy_(orig_vgg.classifier[0].bias.data)
        self.conv7 = nn.Conv2d(4096, 4096, 1, 1, 0)
        self.conv7.weight.data.copy_(
            orig_vgg.classifier[3].weight.data.resize_(4096, 4096, 1, 1))
        self.conv7.bias.data.copy_(orig_vgg.classifier[3].bias.data)

        self.relu6 = nn.ReLU(True)
        self.relu7 = nn.ReLU(True)

        if dropout2d:
            self.dropout6 = nn.Dropout2d(0.5)
            self.dropout7 = nn.Dropout2d(0.5)
        else:
            self.dropout6 = nn.Dropout(0.5)
            self.dropout7 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout6(self.relu6(self.conv6(x)))
        x = self.dropout7(self.relu7(self.conv7(x)))

        return x


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, dropout2d=False):
        super(ResnetDilated, self).__init__()
        self.dropout2d = dropout2d
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        if self.dropout2d:
            self.dropout = nn.Dropout2d(0.5)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.dropout2d:
            x = self.dropout(x)
        return x


# returns multiscale features for QuadNet
class ResnetQuad(nn.Module):
    def __init__(self, orig_resnet):
        super(ResnetQuad, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        s = x

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        s_2 = x

        x = self.maxpool(x)
        x = self.layer1(x)
        s_4 = x

        x = self.layer2(x)
        s_8 = x

        x = self.layer3(x)
        s_16 = x

        x = self.layer4(x)
        s_32 = x

        outs = (s, s_2, s_4, s_8, s_16, s_32)
        return outs


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


# 2 conv with dilation=2, 2 conv with dilation=1, last conv, bilinear upsample
class C5Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(C5Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # convs, dilation=2
        self.conv1 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        self.conv2 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        # convs, dilation=1
        self.conv3 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        self.conv4 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(fc_dim, momentum=0.1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


# pyramid pooling, bilinear upsample
class PSPBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PSPBilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        self.psp = []
        for scale in pool_scales:
            self.psp.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.psp = nn.ModuleList(self.psp)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        psp_out = torch.cat(psp_out, 1)

        x = self.conv_last(psp_out)

        if not (input_size[2] == segSize[0] and input_size[3] == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


# QuadNet based GCN, bilinear upsample
class QuadBilinear(nn.Module):
    def __init__(self, num_class=150, segSize=384,
                 channel_dims = (3, 64, 64, 128, 256, 512), gcn_out = 128,
                 use_softmax=False, extend_depth=False, pixelShuffle=False):
        super(QuadBilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax
        self.extend_depth = extend_depth
        self.pixelShuffle = pixelShuffle

        # channel compression
        if self.extend_depth:
            if self.pixelShuffle:
                self.compress_s_384 = nn.Conv2d(channel_dims[5], gcn_out*9//4, 1, 1, 0, bias=False)
            else:
                self.compress_s_384 = nn.Conv2d(channel_dims[5], gcn_out, 1, 1, 0, bias=False)
            self.compress_s_128 = nn.Conv2d(channel_dims[5], gcn_out, 1, 1, 0, bias=False)
            self.compress_s_64 = nn.Conv2d(channel_dims[5], gcn_out, 1, 1, 0, bias=False)
        self.compress_s_32 = nn.Conv2d(channel_dims[5], gcn_out, 1, 1, 0, bias=False)
        self.compress_s_16 = nn.Conv2d(channel_dims[4], gcn_out, 1, 1, 0, bias=False)
        self.compress_s_8 = nn.Conv2d(channel_dims[3], gcn_out, 1, 1, 0, bias=False)
        self.compress_s_4 = nn.Conv2d(channel_dims[2], gcn_out, 1, 1, 0, bias=False)
        self.compress_s_2 = nn.Conv2d(channel_dims[1], gcn_out, 1, 1, 0, bias=False)
        self.compress_s = nn.Conv2d(channel_dims[0], gcn_out, 1, 1, 0, bias=False)

        # graph convolutions
        if self.pixelShuffle:
            if self.extend_depth:
                self.conv_s_128 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
                self.conv_s_64 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
                self.conv_s_32 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_16 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_8 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_4 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_2 = nn.Conv2d(gcn_out // 4 + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s = nn.Conv2d(gcn_out // 4 + gcn_out, gcn_out, 1, 1, 0, bias=False)
        else:
            if self.extend_depth:
                self.conv_s_128 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
                self.conv_s_64 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
                self.conv_s_32 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_16 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_8 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_4 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s_2 = nn.Conv2d(gcn_out + gcn_out + 4 * gcn_out, gcn_out, 1, 1, 0, bias=False)
            self.conv_s = nn.Conv2d(gcn_out + gcn_out, gcn_out, 1, 1, 0, bias=False)

        # last conv
        if self.extend_depth:
            self.conv_last_s_384 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
            self.conv_last_s_128 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
            self.conv_last_s_64 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
        self.conv_last_s_32 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
        self.conv_last_s_16 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
        self.conv_last_s_8 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
        self.conv_last_s_4 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
        self.conv_last_s_2 = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)
        self.conv_last_s = nn.Conv2d(gcn_out, num_class, 1, 1, 0, bias=False)


    def forward(self, ins, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        (s, s_2, s_4, s_8, s_16, s_32) = ins

        # approximate root nodes with average pooling
        if self.extend_depth:
            av = nn.AvgPool2d(2, 2)
            av3 = nn.AvgPool2d(3, 3)
            s_64 = av(s_32)
            s_128 = av(s_64)
            s_384 = av3(s_128)

        # channel compression for graph convolution features
        if self.extend_depth:
            s_384 = self.compress_s_384(s_384)
            s_128 = self.compress_s_128(s_128)
            s_64 = self.compress_s_64(s_64)
        s_32 = self.compress_s_32(s_32)
        s_16 = self.compress_s_16(s_16)
        s_8 = self.compress_s_8(s_8)
        s_4 = self.compress_s_4(s_4)
        s_2 = self.compress_s_2(s_2)
        s = self.compress_s(s)

        # graph convolutions with pixel shuffle and gather
        if self.pixelShuffle:
            ps = nn.PixelShuffle(2)
            ps3 = nn.PixelShuffle(3)
        else:
            ps = nn.Upsample(scale_factor=2)
            ps3 = nn.Upsample(scale_factor=3)

        if self.extend_depth:
            s_128 = self.conv_s_128(torch.cat([ps3(s_384), s_128, gather(s_64)], 1))
            s_64 = self.conv_s_64(torch.cat([ps(s_128), s_64, gather(s_32)], 1))
            s_32 = self.conv_s_32(torch.cat([ps(s_64), s_32, gather(s_16)], 1))
        s_16 = self.conv_s_16(torch.cat([ps(s_32), s_16, gather(s_8)], 1))
        s_8 = self.conv_s_8(torch.cat([ps(s_16), s_8, gather(s_4)], 1))
        s_4 = self.conv_s_4(torch.cat([ps(s_8), s_4, gather(s_2)], 1))
        s_2 = self.conv_s_2(torch.cat([ps(s_4), s_2, gather(s)], 1))
        s = self.conv_s(torch.cat([ps(s_2), s], 1))

        # label prediction from graph node representations
        if self.extend_depth:
            x_384 = self.conv_last_s_384(s_384)
            x_128 = self.conv_last_s_128(s_128)
            x_64 = self.conv_last_s_64(s_64)
        x_32 = self.conv_last_s_32(s_32)
        x_16 = self.conv_last_s_16(s_16)
        x_8 = self.conv_last_s_8(s_8)
        x_4 = self.conv_last_s_4(s_4)
        x_2 = self.conv_last_s_2(s_2)
        x = self.conv_last_s(s)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)

        if self.extend_depth:
            x_multiscale = (x_2, x_4, x_8, x_16, x_32, x_64, x_128, x_384)
        else:
            x_multiscale = (x_2, x_4, x_8, x_16, x_32)

        return x, x_multiscale