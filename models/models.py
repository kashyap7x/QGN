import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext
from lib.nn import SynchronizedBatchNorm2d
from utils import gather


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label, quad_sup=False):
        _, preds = torch.max(pred, dim=1)
        if quad_sup:
            label_fix = label - 1
            preds_fix = preds - 1
        else:
            label_fix = label
            preds_fix = preds
        valid = ((label_fix >= 0) * (preds_fix >= 0)).long()
        acc_sum = torch.sum(valid * (preds_fix == label_fix).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None, quad_sup=False):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.quad_sup = quad_sup

    def forward(self, feed_dict, *, segSize=None):
        inputs = feed_dict['img_data'].cuda()
        if segSize is None: # training
            labels_orig_scale = feed_dict['seg_label_0'].cuda()             
            labels_scaled = []
            fmap = self.encoder(inputs, return_feature_maps=True)
            
            if self.quad_sup:
                labels_scaled.append(feed_dict['seg_label_1'].cuda())
                labels_scaled.append(feed_dict['seg_label_2'].cuda())
                labels_scaled.append(feed_dict['seg_label_3'].cuda())
                labels_scaled.append(feed_dict['seg_label_4'].cuda())
                labels_scaled.append(feed_dict['seg_label_5'].cuda())
                (pred, pred_quad) = self.decoder(fmap, labels_scaled)
            else:
                pred = self.decoder(fmap)

            loss = self.crit(pred, labels_orig_scale)
            if self.quad_sup:
                for i in range(len(pred_quad)):
                    loss_quad = self.crit(pred_quad[i], labels_scaled[i])
                    loss = loss + loss_quad * (self.deep_sup_scale ** (i + 1))

            acc = self.pixel_acc(pred, labels_orig_scale, self.quad_sup)
            return loss, acc
        else: # inference
            pred = self.decoder(self.encoder(inputs, return_feature_maps=True), segSize=segSize)
            return pred


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet50':
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
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='quadnet',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'quadnet':
            net_decoder = QuadNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                quad_dim=256)
        elif arch.startswith('QGN_'):
            net_decoder = QGN(
                arch=arch,
                num_class=num_class,
                use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


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

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); conv_out.append(x)
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
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

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); conv_out.append(x)
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, context_mode=False,
                 pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax
        self.context_mode = context_mode
        
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Conv2d(fc_dim+len(pool_scales)*512,
                                    num_class, kernel_size=1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            if not self.context_mode:
                x = nn.functional.log_softmax(x, dim=1)
        return x


# GCN-based quadnet
class QuadNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 quad_inplanes=(128,256,512,1024,2048), quad_dim=256):
        super(QuadNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, quad_dim, 1)

        # GCN Module
        self.quad_in = []
        for quad_inplane in quad_inplanes[:-1]:# skip the top layer
            self.quad_in.append(nn.Sequential(
                nn.Conv2d(quad_inplane, quad_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(quad_dim),
                nn.ReLU(inplace=True)
            ))
        self.quad_in = nn.ModuleList(self.quad_in)

        self.quad_gcn = nn.Sequential(
                nn.Conv2d(quad_dim*6, quad_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(quad_dim),
                nn.ReLU(inplace=True)
                )

        self.quad_out = []
        for i in range(len(quad_inplanes) - 1): # skip the bottom layer
            self.quad_out.append(nn.Conv2d(quad_dim, num_class+1, kernel_size=1, bias=False))
        self.quad_out = nn.ModuleList(self.quad_out)


    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(nn.functional.upsample(
                pool_conv(pool_scale(conv5)),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        quad_ins = [f]
        for i in reversed(range(len(conv_out)-1)):
            quad_ins.append(self.quad_in[i](conv_out[i]))

        quad_preds = [self.quad_out[0](f)]
        for i in (range(1,len(conv_out)-1)):
            conv_eq = quad_ins[i]

            conv_minus = quad_ins[i-1]
            conv_minus = nn.functional.upsample(conv_minus, size=conv_eq.size()[2:], mode='bilinear') # top-down branch

            conv_plus = quad_ins[i+1]
            conv_plus = gather(conv_plus)

            gcn_in = torch.cat([conv_eq, conv_minus, conv_plus], 1)
            quad_ins[i] = self.quad_gcn(gcn_in)

            quad_preds.append(self.quad_out[i](quad_ins[i]))

        x = quad_preds[-1]

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x[:,1:,:,:], dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        y = []
        for i in reversed(range(len(quad_preds)-1)):
            y.append(nn.functional.log_softmax(quad_preds[i], dim=1))

        return x, y


# QGN based on sparse transposed ResNet
class QGN(nn.Module):
    def __init__(self, arch, num_class=150, use_softmax=False, use_ppm=False):
        super(QGN, self).__init__()
        self.use_ppm = use_ppm
        if use_ppm:
            self.ppm_context = PPMBilinear(num_class=2048, context_mode=True)
        if arch=='QGN_resnet34':
            self.orig_resnet = resnet.resnet34_transpose_sparse(num_classes=num_class+1)
        elif arch=='QGN_dense_resnet34':
            self.orig_resnet = resnet.resnet34_transpose(num_classes=num_class+1)
        elif arch=='QGN_resnet50':
            self.orig_resnet = resnet.resnet34_transpose_sparse(num_classes=num_class+1)
        elif arch=='QGN_dense_resnet50':
            self.orig_resnet = resnet.resnet34_transpose(num_classes=num_class+1)
        else:
            raise Exception('Architecture undefined!')
        self.use_softmax = use_softmax

    def forward(self, conv_out, labels_scaled=None, segSize=None):
        if self.use_ppm:
            x = self.ppm_context(conv_out)
            conv_out[-1] = x

        quad_preds = self.orig_resnet(conv_out, labels_scaled)
        x = quad_preds[-1]

        if self.use_softmax:  # is True during inference
            masks = [(torch.argmax(out, dim=1)==0).unsqueeze(1).repeat(1,out.shape[1],1,1).type(out.dtype) for out in quad_preds]
            for (i, mask) in enumerate(masks):
                quad_preds[i] = (1 - mask) * quad_preds[i]
                quad_preds[i] = nn.functional.upsample(quad_preds[i], size=segSize, mode='bilinear')
                for j in range(i+1,len(masks)):
                    mask = nn.functional.upsample(mask, scale_factor=2)
                    quad_preds[j] = mask * quad_preds[j]
            x = sum(quad_preds)
            x = nn.functional.softmax(x[:,1:,:,:], dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        y = []
        for i in reversed(range(len(quad_preds)-1)):
            y.append(nn.functional.log_softmax(quad_preds[i], dim=1))
            
        return x, y