import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn

import matplotlib.pyplot as plt


class NeRF_TCNN(nn.Module):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,  # original value is 2
                 hidden_dim=64,  # original value is 64
                 geo_feat_dim=15,  # original value is 15
                 num_layers_color=3,  # original value is 3
                 hidden_dim_color=64,  # original value is 64
                 bound=100,
                 base_resolution=16,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.bound = bound

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.base_resolution = base_resolution

        per_level_scale = np.exp2(np.log2(2048 * bound / base_resolution) / (base_resolution - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def forward(self, input):
        x = input[:, :3]
        d = input[:, 3:]

        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = h

        outputs = torch.cat([color, sigma[..., None]], -1)
        return outputs

class Triplane_Encoding(nn.Module):
    def __init__(self,encoding_config,**kwargs):
        super().__init__(**kwargs)
        self.encoding_config = encoding_config
        self.encoder=torch.nn.ModuleList()
        for i in range(3):
            self.encoder.append(tcnn.Encoding(
                n_input_dims=2,
                encoding_config=encoding_config,
            ))
    def forward(self, input):
        # input: [N, 3]
        # each encoder takes [N, 2]
        # output: [N, 3, 32], sum of 3 encoders
        for i in range(3):
            x = input[:, [i, (i+1)%3]]
            x = self.encoder[i](x)
            if i==0:
                out = x
            else:
                out = torch.clamp(out * x,1e-6)
        return out


class NeRF_TCNN_(nn.Module):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,  # original value is 2
                 hidden_dim=64,  # original value is 64
                 geo_feat_dim=15,  # original value is 15
                 num_layers_color=3,  # original value is 3
                 hidden_dim_color=64,  # original value is 64
                 bound=100,
                 base_resolution=32,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.bound = bound

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.base_resolution = base_resolution

        per_level_scale = np.exp2(np.log2(2048 * bound / base_resolution) / (base_resolution - 1))

        # self.encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": per_level_scale,
        #     },
        # )
        self.encoder = Triplane_Encoding(
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def forward(self, input):
        x = input[:, :3]
        d = input[:, 3:]

        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = h

        outputs = torch.cat([color, sigma[..., None]], -1)
        return outputs


    # def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
    #     num_scenes, _, num_chn, h, w = code.size()
    #     code_viz = code.cpu().numpy()
    #     if not self.flip_z:
    #         code_viz = code_viz[..., ::-1, :]
    #     code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, num_chn * w)
    #     for code_viz_single, scene_name_single in zip(code_viz, scene_name):
    #         plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
    #                    vmin=code_range[0], vmax=code_range[1])

    def sample_from_triplane(self,two_exp_reso,noise_scale=0.):
        # meshgrid of 3 dims in [0,1] with resolution 2**two_exp_reso
        x = torch.linspace(0, 1, 2 ** two_exp_reso)
        x, y = torch.meshgrid(x, x)
        # concate to [N,2]
        x = x.reshape(-1)
        y = y.reshape(-1)
        xy = torch.stack([x, y], dim=-1)

        # add noise, when noise_scale is 1., noise is as large is 1/(2 ** two_exp_reso+1), linearly scaled
        if noise_scale>0.:
            xy += noise_scale*torch.rand_like(xy)*(1./(2 ** two_exp_reso+1))
        feature=[]
        for i in range(3):
            feature.append(self.encoder.encoder[i](xy)) 
        feature = torch.stack(feature, dim=1) # [N,3,32]
        feature = feature.reshape(2 ** two_exp_reso, 2 ** two_exp_reso, 3, -1)
        return feature # [M,M,3,32]

    def total_variance_loss_2d(self,features):
        """
        Compute the 2D total variance loss.
        
        Parameters:
        - features (torch.Tensor): tensor of shape (N, M, M, feature)
        
        Returns:
        - loss (torch.Tensor): total variance loss
        """
        # Assume features is (N, M, M, feature)
        assert len(features.shape) == 4
        assert features.shape[1] == features.shape[2]
        
        # Compute the differences along the x and y dimensions
        diff_x = features[:, 1:, :, :] - features[:, :-1, :, :]
        diff_y = features[:, :, 1:, :] - features[:, :, :-1, :]
        
        # Compute the loss by summing up the absolute differences
        loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
        
        return loss
    
    def total_variance_loss(self,two_exp_reso=None):
        if two_exp_reso is None:
            # choose from [base_resolution//2,base_resolution-1]
            two_exp_reso = np.random.randint(10,12)
            
        feature=self.sample_from_triplane(two_exp_reso) # [M,M,3,32]
        # permute to fit total_variance_loss_2d
        feature = feature.permute(2,0,1,3) # [M,M,3,32] -> [3,M,M,32]
        loss=self.total_variance_loss_2d(feature)
        return loss

    def sparse_loss_2d(self,features,norm='l2'):
        """
        Compute the 2D sparse loss.
        
        Parameters:
        - features (torch.Tensor): tensor of shape (N, M, M, feature)
        
        Returns:
        - loss (torch.Tensor): sparse loss
        """
        # Assume features is (N, M, M, feature)
        assert len(features.shape) == 4
        assert features.shape[1] == features.shape[2]

        if norm=='l2':
            loss = torch.mean(torch.norm(features,dim=-1))
        elif norm=='l1':
            loss = torch.mean(torch.abs(features))
        else:
            raise NotImplementedError
        
        return loss

    def sparse_loss(self,two_exp_reso=None,norm='l2'):
        if two_exp_reso is None:
            two_exp_reso = np.random.randint(10,12)
            
        feature=self.sample_from_triplane(two_exp_reso)
        # permute to fit sparse_loss_2d
        feature = feature.permute(2,0,1,3) # [M,M,3,32] -> [3,M,M,32]
        loss=self.sparse_loss_2d(feature,norm=norm)
        return loss

    def edr_loss(self,two_exp_reso=None,):
        if two_exp_reso is None:
            two_exp_reso = np.random.randint(10,12)
        noise_scale=0.99

        feature=self.sample_from_triplane(two_exp_reso,noise_scale=0.)
        feature_noise=self.sample_from_triplane(two_exp_reso,noise_scale=noise_scale)

        # mse loss between feature and feature_noise
        loss = torch.mean((feature-feature_noise)**2)
        return loss



    




    @torch.no_grad()
    def vis_triplane(self,two_exp_reso,vis_dir,additional_message=[]):
        # meshgrid of 3 dims in [0,1] with resolution 2**two_exp_reso
        feature=self.sample_from_triplane(two_exp_reso) # [M,M,3,32]

        # transform to [0,1] along dim -1
        feature = feature - feature.min(dim=-1, keepdim=True)[0]
        feature = feature / feature.max(dim=-1, keepdim=True)[0]
        # permute and reshape to [3*sqrt(N),sqrt(N)*32]
        feature = feature.permute(2,0,1,3) # [M,M,3,32] -> [3,M,M,32]
        feature = feature.reshape(3*2**two_exp_reso,2**two_exp_reso*32)
        feature = feature.cpu().numpy()
        # add additional message with '_' as separator, note additional_message is a list
        os.makedirs(vis_dir,exist_ok=True)
        save_path=os.path.join(vis_dir,'_'.join(additional_message)+'.png')
        plt.imsave(save_path,feature)
    


