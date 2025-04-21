# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block
from diffusers.models.activations import get_activation
from auv_track_launcher.networks.unet_1d_block import Downsample1d, Upsample1d, Conv1dBlock, ConditionalResidualBlock1D

import einops


@dataclass
class UNet1DConditionOutput(BaseOutput):
    """
    The output of [`UNet1DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, sample_size)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class UNet1DConditionModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        cross_attention_dim=512,
        time_embedding_type: str = "fourier",
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        block_out_channels=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        act_fn="mish",
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(block_out_channels)
        start_dim = block_out_channels[0]

        # Time embedding
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=None,
        )
        
        self.time_embedding = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=block_out_channels[0],
        )
        cond_dim = block_out_channels[0]

        # encoder_global_hidden_states
        self.encoder_hid_proj = nn.Linear(global_cond_dim, cross_attention_dim)
        if cross_attention_dim is not None:
            cond_dim += cross_attention_dim

        # encoder_local_hidden_states   
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv


    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim
    
    def process_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process the encoder hidden states (image features) through a projection layer.
        
        Args:
            encoder_hidden_states (`torch.Tensor`): Image features with shape `(batch_size, feature_dim)`.
            
        Returns:
            `torch.Tensor`: Processed encoder hidden states.
        """
        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        local_encoder_hidden_states: Optional[torch.Tensor] = None,
        # return_dict: bool = True,
    ) -> Union[UNet1DConditionOutput, Tuple]:
        r"""
        The [`UNet1DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the shape `(batch_size, horizon, input_action_dim)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch_size, feature_dim)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~UNet1DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~UNet1DConditionOutput`] or `tuple`:
                If `return_dict` is True, a [`~UNet1DConditionOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        sample = einops.rearrange(sample, 'b h a -> b a h')
        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # Project time embeddings
        t_emb = self.time_proj(timesteps)
        temb = self.time_embedding(t_emb)

        # 2. Global Condition embedding
        # Process encoder hidden states (image features)
        encoder_hidden_states = self.process_encoder_hidden_states(encoder_hidden_states)
        
        # Create conditioning embedding and add to time embedding
        temb = torch.cat([
            temb, encoder_hidden_states
        ], axis=-1)

        # 3. Local Condition embedding
        h_local = list()
        if local_encoder_hidden_states is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, temb)
            h_local.append(x)
            x = resnet2(local_cond, temb)
            h_local.append(x)

        # 4. Down blocks
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, temb)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, temb)
            h.append(x)
            x = downsample(x)

        # down_block_res_samples = ()
        # for downsample_block in self.down_blocks:
        #     sample, res_samples = downsample_block(hidden_states=sample, temb=temb)
        #     down_block_res_samples += res_samples

        # 5. Mid block
        for mid_module in self.mid_modules:
            x = mid_module(x, temb)
        # sample = self.mid_block(sample, temb)

        # 6. Up blocks
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            tmp = h.pop()
            x = torch.cat((x, tmp), dim=1)
            del tmp
            x = resnet(x, temb)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, temb)
            x = upsample(x)

        # for i, upsample_block in enumerate(self.up_blocks):
        #     res_samples = down_block_res_samples[-1:]
        #     down_block_res_samples = down_block_res_samples[:-1]
        #     sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=temb)

        # 6. Output block
        x = self.final_conv(x)
        sample = einops.rearrange(x, 'b t h -> b h t')

        # if self.out_block:
        #     sample = self.out_block(sample, temb)

        # if not return_dict:
        #     return (sample,)

        return UNet1DConditionOutput(sample=sample)
