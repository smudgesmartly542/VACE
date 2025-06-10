# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import logging
from diffusers.configuration_utils import register_to_config
from wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d

# Import GGUF support
try:
    from ...utils.gguf_loader import load_gguf_state_dict, create_gguf_model_config, dequantize_tensor, is_quantized
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logging.warning("GGUF support not available. Install 'gguf' package for quantized model support.")


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
    
    
class VaceWanModel(WanModel):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load model from pretrained path, supporting both regular and GGUF formats
        """
        if os.path.isdir(pretrained_model_name_or_path):
            # Check for GGUF file in directory
            gguf_files = [f for f in os.listdir(pretrained_model_name_or_path) if f.endswith('.gguf')]
            if gguf_files and GGUF_AVAILABLE:
                gguf_path = os.path.join(pretrained_model_name_or_path, gguf_files[0])
                logging.info(f"Found GGUF file: {gguf_path}")
                return cls.from_gguf(gguf_path, **kwargs)
        
        # Fallback to original diffusers loading
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    
    @classmethod
    def from_gguf(cls, gguf_path, device=None, **kwargs):
        """
        Load model from GGUF file
        """
        if not GGUF_AVAILABLE:
            raise ImportError("GGUF support not available. Install 'gguf' package.")
        
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available!")
            device = torch.device("cuda:0")
        
        logging.info(f"Loading VaceWanModel from GGUF: {gguf_path}")
        
        # Create config from GGUF metadata
        config = create_gguf_model_config(gguf_path)
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        # Create model instance with correct config
        model = cls(
            vace_layers=config.get('vace_layers'),
            vace_in_dim=config.get('vace_in_dim'),
            model_type=config.get('model_type', 't2v'),
            patch_size=config.get('patch_size', [1, 2, 2]),
            text_len=config.get('text_len', 512),
            in_dim=config.get('in_dim', 16),
            dim=config.get('dim', 5120),
            ffn_dim=config.get('ffn_dim', 13824),
            freq_dim=config.get('freq_dim', 256),
            text_dim=config.get('text_dim', 4096),
            out_dim=config.get('out_dim', 16),
            num_heads=config.get('num_heads', 40),
            num_layers=config.get('num_layers', 40),
            window_size=config.get('window_size', [-1, -1]),
            qk_norm=config.get('qk_norm', True),
            cross_attn_norm=config.get('cross_attn_norm', True),
            eps=config.get('eps', 1e-6)
        )
        
        # Load GGUF state dict
        state_dict = load_gguf_state_dict(gguf_path, handle_prefix="")
        
        # Load weights with quantization support
        model.load_gguf_state_dict(state_dict, device)
        
        return model.to(device)
    
    def load_gguf_state_dict(self, state_dict, device):
        """
        Load state dict with GGUF tensor handling and on-demand dequantization
        """
        missing_keys = []
        unexpected_keys = []
        
        model_state_dict = self.state_dict()
        
        # Track loading progress
        total_tensors = len(state_dict)
        loaded_tensors = 0
        
        logging.info(f"Loading {total_tensors} tensors from GGUF...")
        
        for key, tensor in state_dict.items():
            if key in model_state_dict:
                # Handle quantized tensors
                param_shape = model_state_dict[key].shape
                if tensor.shape != param_shape:
                    logging.warning(f"Shape mismatch for {key}: expected {param_shape}, got {tensor.shape}")
                    unexpected_keys.append(key)
                    continue
                
                # Get expected dtype
                expected_dtype = model_state_dict[key].dtype
                
                if is_quantized(tensor):
                    # Dequantize tensor
                    logging.debug(f"Dequantizing {key} from {getattr(tensor, 'tensor_type', 'unknown')}")
                    dequantized = dequantize_tensor(tensor, expected_dtype)
                    
                    # Copy the dequantized data
                    with torch.no_grad():
                        model_state_dict[key].copy_(dequantized)
                else:
                    # Regular tensor - just copy
                    with torch.no_grad():
                        model_state_dict[key].copy_(tensor.to(expected_dtype))
                
                loaded_tensors += 1
                if loaded_tensors % 100 == 0:
                    logging.info(f"Loaded {loaded_tensors}/{total_tensors} tensors...")
            else:
                unexpected_keys.append(key)
        
        # Check for missing keys
        for key in model_state_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)
        
        if missing_keys:
            logging.warning(f"Missing keys in GGUF file ({len(missing_keys)}): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in GGUF file ({len(unexpected_keys)}): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
        
        logging.info(f"Successfully loaded {loaded_tensors}/{total_tensors} tensors from GGUF")
        
        return model_state_dict

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        for block in self.vace_blocks:
            c = block(c, **new_kwargs)
        hints = torch.unbind(c)[:-1]
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]