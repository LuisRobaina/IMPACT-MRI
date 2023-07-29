import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .shared_components import ConvBlock, TransposeConvBlock
from argparse import ArgumentParser
from fastmri.pl_modules.mri_module import MriModule


class attention_gate(nn.Module):
    """
    SOURCES: https://arxiv.org/pdf/1804.03999.pdf
                https://idiotdeveloper.com/attention-unet-in-pytorch/
       
        AG is characterised by a set of parameters Θatt containing: linear transformations Wx ∈ Fl×Fint,
        Wg ∈ Fg×Fint, ψ ∈ Fint×1 and bias terms. The linear transformations are computed using
        channel-wise 1x1x1 convolutions for the input tensors. In other contexts [33], this is referred to as
        vector concatenation-based attention, where the concatenated features x and g are linearly mapped
        to a Fint dimensional intermediate space.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0),
            nn.InstanceNorm2d(F_int),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0),
            nn.InstanceNorm2d(F_int),
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,padding=0),
            nn.InstanceNorm2d(1),
        )
        
        self.Sigmoid = nn.Sigmoid()


    def forward(self, g, x):
        #g is the output from convolution
        #x is the skip connection with corresponding dimensions in the downsample
        Wg = self.Wg(g)
        Wx = self.Wx(x)
        alpha = self.relu(Wg + Wx)
        alpha = self.psi(alpha)
        alpha = self.Sigmoid(alpha)
        return x * alpha


class AttentionUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model with Attention Gates

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        
        #setup list of attention gates
        self.attention_gates = nn.ModuleList()
        
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            
            #append attention_gates into list
            self.attention_gates.append(attention_gate(ch, ch, ch//2))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        
        #append one more attention_gate into list
        self.attention_gates.append(attention_gate(ch, ch, ch//2))
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        post_attn_norm_diff = []

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv, attention in zip(self.up_transpose_conv, self.up_conv, self.attention_gates):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            
            #calculate attention using the output from previous layer and skip connection layer
            
            post_attn_downsample_layer = attention(output, downsample_layer)
            post_attn_norm_diff.append(
                post_attn_downsample_layer.norm().item() - downsample_layer.norm().item()
            )
            downsample_layer = post_attn_downsample_layer
            
            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
            
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output, {
            "mean_post_attn_norm_diff": np.mean(post_attn_norm_diff)
        }


class AttnUnetModule(MriModule):
    """
    Attention Unet training module.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.attn_unet = AttentionUnet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    def forward(self, image):
        output, info = self.attn_unet(image.unsqueeze(1))
        self.log("mean_post_attn_norm_diff", info["mean_post_attn_norm_diff"])
        
        return output.squeeze(1), info

    def training_step(self, batch, batch_idx):
        
        output, info = self(batch.image)
        
        loss = F.l1_loss(output, batch.target)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        output, info = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": F.l1_loss(output, batch.target),
        }

    def test_step(self, batch, batch_idx):
        output, info = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
