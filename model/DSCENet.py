
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


class FPE(nn.Module):
    def __init__(self, in_dim=1, prompt_dim=128, hidden_dim=64):
        super(FPE, self).__init__()
        self.prompt_dim = prompt_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),  # 下采样
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),  # 再下采样
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(32)  # 固定到32x32
        )
        self.freq_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.freq_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1),
            nn.Sigmoid()
        )

        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, prompt_dim, 1)
        )

    def forward(self, x):

        feat = self.feature_extractor(x)

        feat_for_fft = self.freq_conv(feat)
        feat_fft = torch.fft.fft2(feat_for_fft.float())
        feat_fft_mag = torch.abs(feat_fft)
        feat_fft_mag = torch.log(1 + feat_fft_mag)

        freq_attn = self.freq_attention(feat_fft_mag)
        weighted_freq = feat_fft_mag * freq_attn

        prompt_vec = self.global_encoder(weighted_freq)
        prompt_vec = prompt_vec.squeeze(-1).squeeze(-1)

        return prompt_vec

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
class PrototypeBankDynamicConv(nn.Module):

    def __init__(self, channels, kernel_size=3, prompt_dim=128,
                 num_prototypes=16, bias=False, init_with_classic=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_prototypes = num_prototypes
        self.prototype_bank = nn.Parameter(
            torch.randn(num_prototypes, kernel_size, kernel_size) * 0.1
        )
        self.weight_net = nn.Sequential(
            nn.Linear(prompt_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channels * num_prototypes)
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x, prompt_freq):
        B, C, H, W = x.shape
        N = self.num_prototypes


        weights = self.weight_net(prompt_freq)  # [B, C*N]
        weights = weights.view(B, C, N)  # [B, C, N]
        weights = F.softmax(weights, dim=-1)  # [B, C, N]

        dynamic_kernels = torch.einsum('bcn,nkl->bckl', weights, self.prototype_bank)

        x_reshaped = x.view(1, B*C, H, W)  # [1, B*C, H, W]
        kernels_reshaped = dynamic_kernels.view(B*C, 1, self.kernel_size, self.kernel_size)
        dynamic_out = F.conv2d(
            x_reshaped,
            kernels_reshaped,
            padding=self.padding,
            groups=B*C
        )
        dynamic_out = dynamic_out.view(B, C, H, W)  # [B, C, H, W]

        out = self.pointwise(dynamic_out)

        return out

##########################################################################
class ALTE(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # ========== 使用核原型库动态卷积 ==========
        self.dwconv3 = PrototypeBankDynamicConv(
            channels=hidden_features * 2,
            kernel_size=3,
            prompt_dim=128,
            num_prototypes=16,
            bias=bias
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, prompt_vec=None):

        x_in = self.project_in(x)
        x3 = self.dwconv3(x_in, prompt_vec)
        x1, x2 = x3.chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.project_out(out)
        return out

##########################################################################
class AGFA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AGFA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = PrototypeBankDynamicConv(
            channels=dim * 3,
            kernel_size=3,
            prompt_dim=128,
            num_prototypes=16,
            bias=bias
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, prompt_vec=None):
        b, c, h, w = x.shape
        qkv_input = self.qkv(x)
        qkv = self.qkv_dwconv(qkv_input, prompt_vec)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class AGLFE(nn.Module):
    def __init__(self, dim, num_heads, bias=False, prompt_dim=128):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.AGFA = AGFA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.ALTE = ALTE(dim, ffn_expansion_factor=2.66, bias=bias)

    def forward(self, x, prompt_vec=None):
        x = x + self.AGFA(self.norm1(x), prompt_vec=prompt_vec)
        x = x + self.ALTE(self.norm2(x), prompt_vec=prompt_vec)
        return x


##########################################################################
## Patch Embedding & Up/Downsample
##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
class DSCENet(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=48,
                 num_blocks=[4,6,6,8], heads=[1,2,4,8], bias=False,
                 num_modalities: int = 3):
        super().__init__()


        self.freq_prompt = FPE(
            in_dim=inp_channels,
            prompt_dim=128,
            hidden_dim=64
        )


        self.num_modalities = num_modalities
        self.prompt_classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_modalities)
        )

        # Patch embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 编码器
        self.encoder_level1 = nn.ModuleList([
            AGLFE(dim, heads[0], prompt_dim=128) for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.ModuleList([
            AGLFE(dim*2, heads[1], prompt_dim=128) for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim*2)

        self.encoder_level3 = nn.ModuleList([
            AGLFE(dim*4, heads[2], prompt_dim=128) for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim*4)

        # 潜在层
        self.latent = nn.ModuleList([
            AGLFE(dim*8, heads[3], prompt_dim=128) for _ in range(num_blocks[3])
        ])

        # 解码器
        self.up4_3 = Upsample(dim*8)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            AGLFE(dim*4, heads[2], prompt_dim=128) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            AGLFE(dim*2, heads[1], prompt_dim=128) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim*2)
        self.decoder_level1 = nn.ModuleList([
            AGLFE(dim*2, heads[0], prompt_dim=128) for _ in range(num_blocks[0])
        ])

        # 细化层
        self.refinement = nn.ModuleList([
            AGLFE(dim*2, heads[0], prompt_dim=128) for _ in range(num_blocks[0])
        ])

        # 输出层
        self.output = nn.Conv2d(dim*2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _forward_blocks(self, blocks, x, prompt_vec):
        for blk in blocks:
            x = blk(x, prompt_vec=prompt_vec)
        return x

    def forward(self, inp_img):

        prompt_vec = self.freq_prompt(inp_img)
        prompt_logits = self.prompt_classifier(prompt_vec)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self._forward_blocks(self.encoder_level1, inp_enc_level1, prompt_vec)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self._forward_blocks(self.encoder_level2, inp_enc_level2, prompt_vec)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self._forward_blocks(self.encoder_level3, inp_enc_level3, prompt_vec)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self._forward_blocks(self.latent, inp_enc_level4, prompt_vec)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self._forward_blocks(self.decoder_level3, inp_dec_level3, prompt_vec)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self._forward_blocks(self.decoder_level2, inp_dec_level2, prompt_vec)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self._forward_blocks(self.decoder_level1, inp_dec_level1, prompt_vec)

        out_dec_level1 = self._forward_blocks(self.refinement, out_dec_level1, prompt_vec)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        if self.training:
            return out_dec_level1, prompt_logits
        else:
            return out_dec_level1


