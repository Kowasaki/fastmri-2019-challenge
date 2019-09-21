import torch

from torch import nn
from torch.nn import functional as F

from models.unet.unet_model import UnetAuto


class FeatureLoss(nn.Module):
    def __init__(self, checkpoint_pth, args, feat_weight=1):
        super().__init__()

        self.w = feat_weight
        self.model = UnetAuto(
            in_chans=1,
            out_chans=1,
            chans=64,
            num_pool_layers=args.num_pools,
            drop_prob=args.drop_prob
        ).to(args.device)

        if args.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(checkpoint_pth)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.model.eval()

        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss



    def forward(self, input, target):
        input = input.unsqueeze(1)
        target = target.unsqueeze(1)

        _, in_intermediates = self.model(input, encode_only=True)
        _, out_intermediates = self.model(target, encode_only=True)

        in_intermediates = in_intermediates[3:6]
        out_intermediates = out_intermediates[3:6]

        self.feat_losses = [self.l1_loss(input,target)]
        self.feat_losses += [self.l1_loss(f_in, f_out)*self.w for f_in, f_out in zip(in_intermediates, out_intermediates)]
        
        return sum(self.feat_losses)
    
