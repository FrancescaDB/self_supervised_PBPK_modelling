import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Enforce determinism
random_seed = 1 
torch.use_deterministic_algorithms(True) 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck, config):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck
        self.use_batch_norm = config["batch_norm"]
        max_pool_time = config["max_pool_time"]

        self.kernel = {"stride": 1, "padding": (1, 1, 1), "kernel_size": (3, 3, 3)}

        self.conv3d_in = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel["kernel_size"], stride=self.kernel["stride"], padding=self.kernel["padding"])
        self.conv3d = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel["kernel_size"], stride=self.kernel["stride"], padding=self.kernel["padding"])
        self.batch_norm = nn.BatchNorm3d(num_features=self.out_channels)
        
        if max_pool_time:       self.pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        else:                   self.pooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2, padding=0)

        if config["activation"] == "ELU":     
            self.activation = nn.ELU()
        elif config["activation"] == "ReLU":        
            self.activation = nn.ReLU()
        else:
            print("*** Invalid config['activation']: using ELU ***")
            self.activation = nn.ELU()

    def forward(self, x):
        i = x
        o1 = self.conv3d_in(i)
        if self.use_batch_norm:
            o1 = self.batch_norm(o1)
        o2 = self.activation(o1)

        o3 = self.conv3d(o2)
        if self.use_batch_norm:
            o3 = self.batch_norm(o3)
        o4 = self.activation(o3)

        if not self.bottleneck:
            skip = torch.cat((i, o4), dim=1)               
            down_sampling_features = skip                 # Short+Long skip connection
            out = self.pooling(o4)
        else: 
            out = o4
            down_sampling_features = None
        return out, down_sampling_features

class Encoder(nn.Module):
    def __init__(self, in_channels, config):
        # # # print("This is the encoder")
        super(Encoder, self).__init__()
        self.root_feat_maps = 8
        self.use_batch_norm = config["batch_norm"]
        model_depth = config["model_depth"]

        self.module_dict = nn.ModuleDict()
        current_in_channels = in_channels
        for depth in range(model_depth):
            current_out_channels = 2 ** (depth) * self.root_feat_maps
            encoder_block = EncoderBlock(current_in_channels, current_out_channels, bottleneck=False, config=config)
            self.module_dict["encoder_block_{}".format(depth+1)] = encoder_block
            current_in_channels = current_out_channels 
        current_out_channels = 2 ** (model_depth) * self.root_feat_maps
        bottle_neck = EncoderBlock(current_in_channels, current_out_channels, bottleneck=True, config=config)
        self.module_dict["bottle_neck"] = bottle_neck

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            x, down_f = op(x)
            down_sampling_features.append(down_f)

        return x, down_sampling_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, final_layer, config):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.final_layer = final_layer
        self.use_batch_norm = config["batch_norm"]
        
        self.kernel = {"stride": 1, "padding": (1, 1, 1), "kernel_size": (3, 3, 3)}
        self.upkernel = {"stride": (2, 2, 2), "padding": (1, 1, 1), "kernel_size": (3, 3, 3), "output_padding": (1, 1, 1)}
        self.final_kernel = {"stride": 1, "padding": 1, "kernel_size": 3}

        self.upconv3d = nn.ConvTranspose3d(in_channels=self.in_channels, out_channels=self.in_channels, 
                                                    kernel_size=self.upkernel["kernel_size"], stride=self.upkernel["stride"], 
                                                    padding=self.upkernel["padding"], output_padding=self.upkernel["output_padding"])
        self.conv3d_skip = nn.Conv3d(in_channels=self.in_channels+self.skip_channels, out_channels=self.out_channels, 
                                            kernel_size=self.kernel["kernel_size"], stride=self.kernel["stride"], padding=self.kernel["padding"])
        self.conv3d = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, 
                                            kernel_size=self.kernel["kernel_size"], stride=self.kernel["stride"], padding=self.kernel["padding"])
        
        self.batch_norm = nn.BatchNorm3d(num_features=self.out_channels)
        self.batch_norm_skip = nn.BatchNorm3d(num_features=self.in_channels+(self.out_channels))
        
        if config["activation"] == "ELU":     
            self.activation = nn.ELU()
        elif config["activation"] == "ReLU":        
            self.activation = nn.ReLU()
        else:
            print("*** Invalid config['activation']: using ELU ***")
            self.activation = nn.ELU()

        if not self.final_layer is None:
            self.final_conv_1 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.final_layer, kernel_size=self.final_kernel["kernel_size"], stride=self.final_kernel["stride"], padding=self.final_kernel["padding"])                      
            self.final_conv_2 = nn.Conv3d(in_channels=self.final_layer, out_channels=self.final_layer, kernel_size=(64, 1, 1), stride=1, padding=0)

    def forward(self, x, down_f):
        i = x
        o1 = self.upconv3d(i)
        o2_skip = torch.cat((o1, down_f), dim=1)             # Long skip connection 
        o3 = self.conv3d_skip(o2_skip)
        if self.use_batch_norm:
            o3 = self.batch_norm(o3)
        o4 = self.activation(o3)

        o5 = self.conv3d(o4)
        if self.use_batch_norm:
            o5 = self.batch_norm(o5)
        out = self.activation(o5)

        if not self.final_layer is None:
            out = self.final_conv_1(out)
            out = self.final_conv_2(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(Decoder, self).__init__()
        self.root_feat_maps = 8
        self.use_batch_norm = config["batch_norm"]
        model_depth = config["model_depth"]

        short_skip_channels = list()
        long_skip_channels = list()
        current_in_channels = in_channels
        for depth in range(model_depth):
            short_skip_channels.append(current_in_channels)
            current_out_channels = 2 ** (depth) * self.root_feat_maps
            long_skip_channels.append(current_in_channels+current_out_channels)
            current_in_channels = current_out_channels

        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth, 0, -1):
            current_in_channels = 2 ** (depth) * self.root_feat_maps
            current_out_channels = 2 ** (depth-1) * self.root_feat_maps
            current_skip_channels = long_skip_channels[depth-1]
            if depth == 1:
                decoder_block = DecoderBlock(current_in_channels, current_out_channels, current_skip_channels, final_layer=out_channels, config=config)
            else:
                decoder_block = DecoderBlock(current_in_channels, current_out_channels, current_skip_channels, final_layer=None, config=config)
            self.module_dict["decoder_block_{}".format(depth)] = decoder_block

    def forward(self, x, down_sampling_features):
        op_dict = list(self.module_dict.items())
        down_sampling_features = down_sampling_features[0:len(op_dict)]
        for i in range(len(op_dict)):
            k, op = op_dict[i]
            down_f = down_sampling_features[len(op_dict)-1-i]
            x = op(x, down_f)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(UNet, self).__init__()
        
        print("*** THIS IS THE CONVENTIONAL UNET ***")

        self.encoder = Encoder(in_channels, config)
        self.decoder = Decoder(in_channels, out_channels, config)

        return

    def forward(self, x):
        y, h = self.encoder(x)
        z = self.decoder(y, h)
        return z

