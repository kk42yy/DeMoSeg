import torch
import numpy as np
from torch import nn
from training.network.ConvBlock import ConvDropoutNormNonlin
from training.network.CSA_CSABlock import CSADropoutNormNonlin, SplitConvCSA

class DeMoSeg(nn.Module):
    def __init__(self,
                 input_channels=4,
                 base_num_features=32,
                 num_classes=4,
                 num_pool=5,
                 do_ds=True,
                 modality=14
                ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.modality = modality
        self.num_pool = num_pool
        self.do_ds = do_ds # Deep Supervision

        self.conv_blocks_localization = [] # Up
        self.conv_blocks_context = [] # Down
        self.tu = [] # ConvTranspose
        self.seg_outputs = [] # Segment Head

        self.inference_apply_nonlin = nn.Softmax(dim=1)
        self.training_apply_nonlin = nn.Identity()

        self.features_per_stage=[min(base_num_features * 2 ** i, 320) for i in range(num_pool+1)]
        self.conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.conv_kwargs_down = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True}
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.weightInitializer = InitWeights_He(1e-2)

        # 1. building Encoder
        self.conv_blocks_context.append(SplitConvCSA(self.features_per_stage[0])) # FD + CSSA + RCR
        for in_fea, out_fea in zip(self.features_per_stage[:-1], self.features_per_stage[1:]):
            self.conv_blocks_context.append(nn.Sequential(
                ConvDropoutNormNonlin(in_fea, out_fea, conv_kwargs=self.conv_kwargs_down),
                CSADropoutNormNonlin(out_fea, sort_small_first=True)
            ))

        # 2. building TransposeConv and Decoder
        for in_fea, up_out_fea in zip(self.features_per_stage[::-1][:-1], self.features_per_stage[::-1][1:]):
            self.tu.append(
                nn.ConvTranspose3d(in_fea, up_out_fea, kernel_size=2, stride=2, bias=False)
            )
            self.conv_blocks_localization.append(nn.Sequential(
                ConvDropoutNormNonlin(up_out_fea+up_out_fea, up_out_fea, conv_kwargs=self.conv_kwargs),
                CSADropoutNormNonlin(up_out_fea, sort_small_first=True)
            ))

        # 3. building Segment Head
        for fea in self.features_per_stage[:-1][::-1]:
            self.seg_outputs.append(nn.Conv3d(fea, self.num_classes, 1, 1, bias=False))

        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.apply(self.weightInitializer)

    def forward(self, x):
        if self.training:
            missing_case = torch.randint(0,15,(x.shape[0],)).cpu().numpy()
        else:
            missing_case = torch.zeros(x.shape[0], dtype=torch.int64).fill_(self.modality).cpu().numpy()
        x = self.MissingSituationGeneratemorebatch(x, missing_case)
        skips = []
        seg_outputs = []

        # 1. Encoder
        for i, layer_enc in enumerate(self.conv_blocks_context):
            x = layer_enc(x, missing_case) if i == 0 else layer_enc(x)
            skips.append(x)

        skips.pop()
        # 2. Decoder
        for transpose, layer_dec, seg_head in zip(self.tu, self.conv_blocks_localization, self.seg_outputs):
            x = layer_dec(torch.cat([transpose(x), skips.pop()], dim=1))
            seg_outputs.append(self.training_apply_nonlin(seg_head(x)))

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]

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
    
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
