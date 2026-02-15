import torch
from torch import nn
from training.network.ConvBlock import ConvDropoutNormNonlin

class CSA(nn.Module):
    def __init__(self, inc=512, kernel_size=3, ratio=0.25, sort_small_first=False):
        super(CSA, self).__init__()
        self.inconv = nn.Conv3d(inc, inc, kernel_size, 1, 1)
        self.innorm = nn.InstanceNorm3d(inc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.ch_order = nn.Sequential(
            nn.Linear(inc, int(inc*ratio)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(int(inc*ratio), inc),
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
    def __init__(self, channels,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 sort_small_first=False):
        super(CSADropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op

        self.conv = CSA(channels, kernel_size=3, sort_small_first=sort_small_first)
        self.instnorm = self.norm_op(channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))
    
class SplitConvCSA_noKL(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()

        self.num_features = num_features
        self.t1_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1_csa = CSADropoutNormNonlin(num_features, sort_small_first=True)
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
        return newx
    
class SplitConvCSA(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.output_channels = num_features
        self.num_features = num_features
        self.t1_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1_csa = CSADropoutNormNonlin(num_features, sort_small_first=True)
        self.t1ce_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1ce_csa = CSADropoutNormNonlin(num_features)
        self.t2_conv = ConvDropoutNormNonlin(1, num_features)
        self.t2_csa = CSADropoutNormNonlin(num_features)
        self.t2flare_conv = ConvDropoutNormNonlin(1, num_features)
        self.t2flare_csa = CSADropoutNormNonlin(num_features)
        self.klloss = torch.nn.KLDivLoss()

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
        # t1: torch.Tensor = self.t1_csa(self.t1_conv(t1))
        # t1ce: torch.Tensor = self.t1ce_csa(self.t1ce_conv(t1ce))
        # t2: torch.Tensor = self.t2_csa(self.t2_conv(t2))
        # t2flare: torch.Tensor = self.t2flare_csa(self.t2flare_conv(t2flare))

        t1_ = self.t1_conv(t1)
        t1: torch.Tensor = self.t1_csa(t1_)

        t1ce_ = self.t1ce_conv(t1ce)
        t1ce: torch.Tensor = self.t1ce_csa(t1ce_)

        t2_ = self.t2_conv(t2)
        t2: torch.Tensor = self.t2_csa(t2_)

        t2flare_ = self.t2flare_conv(t2flare)
        t2flare: torch.Tensor = self.t2flare_csa(t2flare_)
        
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
        if self.training:
            return newx, self.kl_loss_calculate(t1_, t1ce_, t2_, t2flare_, missing_index)
        return newx

    def kl_loss_calculate(self, t1: torch.Tensor, ce: torch.Tensor, t2: torch.Tensor, fl: torch.Tensor, missing_index):
        klloss = self.missing_klloss(
            t1[0].unsqueeze(0), 
            ce[0].unsqueeze(0), 
            t2[0].unsqueeze(0), 
            fl[0].unsqueeze(0),
              missing_index[0]
        )
        for b in range(1, len(missing_index)):
            klloss += self.missing_klloss(
                t1[b].unsqueeze(0), 
                ce[b].unsqueeze(0), 
                t2[b].unsqueeze(0), 
                fl[b].unsqueeze(0), 
                missing_index[b]
            )
        return klloss * 0.5

    def missing_klloss(self, t1, ce, t2, fl, missing_index):
        batch_klloss = 0
        C1, C2, C3, C4 = [self.num_features * i // 4 for i in range(1,5)]

        # t1: t1, ce, t2, fl
        # ce: ce, t1, fl, t2
        # t2: t2, fl, t1, ce
        # fl: fl, t2, ce, t1

        if missing_index <= 3: # t1, t1ce, t2, flair
            pass        
        elif missing_index == 4: # t1 t1ce
            batch_klloss += self.kl_loss(t1[:,C1:C2], ce[:,:C1]) + self.kl_loss(ce[:,C1:C2], t1[:,:C1])
        elif missing_index == 5: # t1 t2
            batch_klloss += self.kl_loss(t1[:,C2:C3], t2[:,:C1]) + self.kl_loss(t2[:,C2:C3], t1[:,:C1])
        elif missing_index == 6: # t1 flair
            batch_klloss += self.kl_loss(t1[:,C3:C4], fl[:,:C1]) + self.kl_loss(fl[:,C3:C4], t1[:,:C1])
        elif missing_index == 7: # t1ce t2
            batch_klloss += self.kl_loss(ce[:,C3:C4], t2[:,:C1]) + self.kl_loss(t2[:,C3:C4], ce[:,:C1])
        elif missing_index == 8: # t1ce flair
            batch_klloss += self.kl_loss(ce[:,C2:C3], fl[:,:C1]) + self.kl_loss(fl[:,C2:C3], ce[:,:C1])
        elif missing_index == 9: # t2 flair
            batch_klloss += self.kl_loss(t2[:,C1:C2], fl[:,:C1]) + self.kl_loss(fl[:,C1:C2], t2[:,:C1])
        elif missing_index == 10: # t1 t1ce t2
            batch_klloss += self.kl_loss(t1[:,C1:C2], ce[:,:C1]) + self.kl_loss(t1[:,C2:C3], t2[:,:C1]) + \
                            self.kl_loss(ce[:,C1:C2], t1[:,:C1]) + self.kl_loss(ce[:,C3:C4], t2[:,:C1]) + \
                            self.kl_loss(t2[:,C2:C3], t1[:,:C1]) + self.kl_loss(t2[:,C3:C4], ce[:,:C1])
        elif missing_index == 11: # t1 t1ce flair
            batch_klloss += self.kl_loss(t1[:,C1:C2], ce[:,:C1]) + self.kl_loss(t1[:,C3:C4], fl[:,:C1]) + \
                            self.kl_loss(ce[:,C1:C2], t1[:,:C1]) + self.kl_loss(ce[:,C2:C3], fl[:,:C1]) + \
                            self.kl_loss(fl[:,C2:C3], ce[:,:C1]) + self.kl_loss(fl[:,C3:C4], t1[:,:C1])
        elif missing_index == 12: # t1 t2 flair
            batch_klloss += self.kl_loss(t1[:,C2:C3], t2[:,:C1]) + self.kl_loss(t1[:,C3:C4], fl[:,:C1]) + \
                            self.kl_loss(t2[:,C1:C2], fl[:,:C1]) + self.kl_loss(t2[:,C2:C3], t1[:,:C1]) + \
                            self.kl_loss(fl[:,C1:C2], t2[:,:C1]) + self.kl_loss(fl[:,C3:C4], t1[:,:C1])
        elif missing_index == 13: # t1ce t2 flair
            batch_klloss += self.kl_loss(ce[:,C2:C3], fl[:,:C1]) + self.kl_loss(ce[:,C3:C4], t2[:,:C1]) + \
                            self.kl_loss(t2[:,C1:C2], fl[:,:C1]) + self.kl_loss(t2[:,C3:C4], ce[:,:C1]) + \
                            self.kl_loss(fl[:,C1:C2], t2[:,:C1]) + self.kl_loss(fl[:,C2:C3], ce[:,:C1])
        elif missing_index == 14: # t1 t1ce t2 flair
            batch_klloss += self.kl_loss(t1[:,C1:C2], ce[:,:C1]) + self.kl_loss(t1[:,C2:C3], t2[:,:C1]) + self.kl_loss(t1[:,C3:C4], fl[:,:C1]) + \
                            self.kl_loss(ce[:,C1:C2], t1[:,:C1]) + self.kl_loss(ce[:,C2:C3], fl[:,:C1]) + self.kl_loss(ce[:,C3:C4], t2[:,:C1]) + \
                            self.kl_loss(t2[:,C1:C2], fl[:,:C1]) + self.kl_loss(t2[:,C2:C3], t1[:,:C1]) + self.kl_loss(t2[:,C3:C4], ce[:,:C1]) + \
                            self.kl_loss(fl[:,C1:C2], t2[:,:C1]) + self.kl_loss(fl[:,C2:C3], ce[:,:C1]) + self.kl_loss(fl[:,C3:C4], t1[:,:C1])
        else:
            raise IndexError
        return batch_klloss

    def kl_loss(self, x, y):
        # x, y: [1, C, D, H, W]
        T = 1
        x = torch.nn.functional.log_softmax(x / T, dim=1)
        y = torch.nn.functional.softmax(y / T, dim=1)
        return self.klloss(x, y) * T * T
