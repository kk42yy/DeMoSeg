#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunetv2.utilities.helpers import softmax_helper_dim1 as softmax_helper
from torch import nn
import torch
import numpy as np
import torch.nn.functional

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if input_channels == output_channels and conv_kwargs['stride'] == 1:
            self.conv = CSA(input_channels, sort_small_first=True)
        else:
            self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        
        # self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class CSA(nn.Module):
    def __init__(self, inc=512, kernel_size=3, ratio=0.25, sort_small_first=False):
        super(CSA, self).__init__()
        self.inconv = nn.Conv3d(inc, inc, kernel_size, 1, 1)
        self.innorm = nn.InstanceNorm3d(inc)
        # self.gelu = nn.GELU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.avg = nn.AdaptiveAvgPool3d(1)
        self.ch_order = nn.Sequential(
            nn.Linear(inc, int(inc*ratio)),
            # nn.GELU(),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(int(inc*ratio), inc),
            # nn.GELU(),
            nn.Sigmoid()
        )

        self.sort_small_first = sort_small_first

    def forward(self, x:torch.Tensor):
        b,c,d,h,w = x.size()


        x = self.inconv(x)
        x = self.lrelu(self.innorm(x))
        x_res = x
        ch_order = torch.argsort(self.ch_order(self.avg(x).view(b,c)), descending=not self.sort_small_first)
        
        return x_res + self.exchange(x, ch_order)


    @staticmethod
    def exchange(x: torch.Tensor, channel_order: torch.Tensor):
        b,c,d,h,w = x.size()
        new_x = []
        for batch in range(b):
            batch_order = channel_order[batch]
            new_x.append(x[batch][batch_order].unsqueeze(0))
        return torch.vstack(new_x)
    
class CSADropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, channels,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 csa_sort_small_first=False):
        super(CSADropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op

        self.conv = CSA(channels, kernel_size=3, sort_small_first=csa_sort_small_first)
        self.instnorm = self.norm_op(channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))
    
class SplitConvCSA(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.output_channels = num_features
        self.num_features = num_features
        self.t1_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1_csa = CSADropoutNormNonlin(num_features, csa_sort_small_first=True)
        self.t1ce_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1ce_csa = CSADropoutNormNonlin(num_features)
        self.t2_conv = ConvDropoutNormNonlin(1, num_features)
        self.t2_csa = CSADropoutNormNonlin(num_features)
        self.t2flare_conv = ConvDropoutNormNonlin(1, num_features)
        self.t2flare_csa = CSADropoutNormNonlin(num_features)

    def feature_chosen(self, t1___, t1ce_, t2___, flair, missing_index):
        '''
        t1 <-> t2, t1ce <-> flair
        '''
        if missing_index == 0:   # t1
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1___[:,self.num_features*2//4:self.num_features*3//4,...],
                                t1___[:,self.num_features*3//4:self.num_features*4//4,...]], dim=1)
        elif missing_index == 1: # t1ce
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*3//4:self.num_features*4//4,...],
                                t1ce_[:,self.num_features*2//4:self.num_features*3//4,...]], dim=1)
        elif missing_index == 2: # t2
            batch1 = torch.cat([t2___[:,self.num_features*2//4:self.num_features*3//4,...],
                                t2___[:,self.num_features*3//4:self.num_features*4//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 3: # flair
            batch1 = torch.cat([flair[:,self.num_features*3//4:self.num_features*4//4,...],
                                flair[:,self.num_features*2//4:self.num_features*3//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 4: # t1 t1ce
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*2//4:self.num_features*3//4,...],
                                t1ce_[:,self.num_features*2//4:self.num_features*3//4,...]], dim=1)
        elif missing_index == 5: # t1 t2
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 6: # t1 flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 7: # t1ce t2
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 8: # t1ce flair
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 9: # t2 flair
            batch1 = torch.cat([t2___[:,self.num_features*2//4:self.num_features*3//4,...],
                                flair[:,self.num_features*2//4:self.num_features*3//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 10: # t1 t1ce t2
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 11: # t1 t1ce flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 12: # t1 t2 flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 13: # t1ce t2 flair
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 14: # t1 t1ce t2 flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        else:
            raise IndexError
        
        return batch1
    
   
    def forward(self, x, missing_index):
        t1, t1ce, t2, t2flare = torch.split(x, 1, dim=1)
        t1: torch.Tensor = self.t1_csa(self.t1_conv(t1))
        t1ce: torch.Tensor = self.t1ce_csa(self.t1ce_conv(t1ce))
        t2: torch.Tensor = self.t2_csa(self.t2_conv(t2))
        t2flare: torch.Tensor = self.t2flare_csa(self.t2flare_conv(t2flare))

        newx = self.feature_chosen(
                t1[0,...].unsqueeze(0),
                t1ce[0,...].unsqueeze(0), 
                t2[0,...].unsqueeze(0), 
                t2flare[0,...].unsqueeze(0),
                missing_index[0]
            )
        for b, missnum in enumerate(missing_index[1:], start=1):
            newx = torch.cat([newx, 
                              self.feature_chosen(
                                t1[b,...].unsqueeze(0),
                                t1ce[b,...].unsqueeze(0), 
                                t2[b,...].unsqueeze(0), 
                                t2flare[b,...].unsqueeze(0),
                                missnum)
                              ], dim=0)
        return newx, (t1, t1ce, t2, t2flare)

class DeMoSeg(nn.Module):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False,
                 modality=14):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(DeMoSeg, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self.deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.modality = modality

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            if d == 0:
                self.conv_blocks_context.append(
                    SplitConvCSA(output_features)
                )
                input_features = output_features
                output_features = int(np.round(output_features * feat_map_mul_on_downscale))
                output_features = min(output_features, self.max_num_features)
                continue

            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

    def forward(self, x, missing_case=None):
        if self.training:
            if missing_case is None: # don't use klloss, i.e., other trainer
                missing_case = torch.randint(0,15,(x.shape[0],)).cpu().numpy()
                self.ReturnKLLoss = False
            else:
                self.ReturnKLLoss = True
        else:
            missing_case = torch.zeros(x.shape[0], dtype=torch.int64).fill_(self.modality).cpu().numpy()
            self.ReturnKLLoss = False
        
        # 0. Simulate Modality Missing situation
        if self.training:
            x = self.MissingSituationGeneratemorebatch(x, missing_case)
        else:
            x = self.MissingSituationGeneratemorebatch(x, missing_case)

        skips = []
        seg_outputs = []
        KLLoss_Feature = []
        for d in range(len(self.conv_blocks_context) - 1):
            if d == 0:
                if self.ReturnKLLoss:
                    x, KLLoss_Feature = self.conv_blocks_context[d](x, missing_case)
                else:
                    x, _ = self.conv_blocks_context[d](x, missing_case)

            else:
                x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.deep_supervision and self.do_ds:
            if self.ReturnKLLoss:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                                zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), KLLoss_Feature
            else:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                                zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        
    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    @staticmethod
    def MissingSituationGeneratemorebatch(x: torch.Tensor, missing_cases_index: np.ndarray):
        """
        Our modality order is t1, t1ce, t2, flare
        0: t1
        1: t1ce
        2: t2
        3: flare
        4: t1, t1ce
        5: t1, t2
        6: t1, flare
        7: t1ce, t2
        8: t1ce, flare
        9: t2, flare
        10: t1, t1ce, t2
        11: t1, t1ce, flare
        12: t1, t2, flare
        13: t1ce, t2, flare
        14: t1, t1ce, t2, flare
        """
        missing_situation_dict = {
            0: [1,0,0,0],
            1: [0,1,0,0],
            2: [0,0,1,0],
            3: [0,0,0,1],
            4: [1,1,0,0],
            5: [1,0,1,0],
            6: [1,0,0,1],
            7: [0,1,1,0],
            8: [0,1,0,1],
            9: [0,0,1,1],
            10: [1,1,1,0],
            11: [1,1,0,1],
            12: [1,0,1,1],
            13: [0,1,1,1],
            14: [1,1,1,1]
        }
        random_miss = [missing_situation_dict[i] for i in missing_cases_index]
        random_miss = torch.from_numpy(np.array(random_miss)).to(x.device).view(x.shape[0], 4, 1, 1, 1)
        x = x * random_miss
        return x
        
if __name__ == "__main__":
    network = DeMoSeg(
        input_channels=4,
        base_num_features=32,
        num_classes=3,
        num_pool=5,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs = {'eps': 1e-5, 'affine': True},
        dropout_op_kwargs = {'p': 0, 'inplace': True},
        final_nonlin=lambda x:x,
        pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        convolutional_pooling=True,
        convolutional_upsampling=True
    )
    print(network)
    # x = torch.randn(2,4,128,128,128)
    # print(network(x)[0].shape)