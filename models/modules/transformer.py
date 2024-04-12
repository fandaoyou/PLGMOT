from functools import partial

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_c=256, embed_dim=256, norm_layer=None):
        super().__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(1, 1))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        bs, num_pos, c, h, w = x.shape
        # temp_x: [bs*num_pos, 256, 7, 7]
        x = x.reshape(bs*num_pos, c, h, w)

        # proj: [bs*num_pos, 256, 7, 7] -> [bs*num_pos, 256, 7, 7]
        x = self.proj(x)

        # flatten: [bs*num_pos, 256, 7, 7] -> [bs, 256, num_pos*49]
        # transpose: [bs, 256, num_pos] -> [B, num_pos, C]
        x = x.reshape(bs, c, num_pos*h*w)
        x = x.transpose(1, 2)

        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_pos, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_pos, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_pos, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_pos, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_pos, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_pos]
        # @: multiply -> [batch_size, num_heads, num_pos, num_pos]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_pos, embed_dim_per_head]
        # transpose: -> [batch_size, num_pos, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_pos, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureReconstructTransformer(nn.Module):
    def __init__(self, in_c=3, embed_dim=768, depth=12, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0., attn_drop_ratio=0., norm_layer=None, act_layer=None):
        super(FeatureReconstructTransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(in_c=in_c, embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Weight init
        self.apply(_init_vit_weights)

    def forward(self, x):
        # [bs, num_pos, 256, 7, 7] -> [bs, num_pos, 256]
        bs, num_pos, c, h, w = x.shape
        x_embed = self.patch_embed(x) # []

        x_embed = self.pos_drop(x_embed)
        x_embed = self.blocks(x_embed)
        x_embed = self.norm(x_embed)
        x_embed = x_embed.reshape(bs, num_pos, c, h, w)
        out = x_embed.mean(dim=1)
        return out


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == "__main__":
    batch_size = 2
    num_props = 12
    channels = 256

    props_feat = torch.rand(batch_size, num_props, channels, 7, 7).cuda()

    FRT = FeatureReconstructTransformer(in_c=256, embed_dim=256)
    FRT = FRT.cuda()

    out = FRT(props_feat)
    # >> following lines show the structure of FRT as well as the input and output shapes
    print(FRT)

    print('-' * 50)
    print('Input shape: \n\
            Proposal: {}\n'.format(
        ' x '.join(list(map(str, props_feat.shape))),
    ))
    print('Output shape: \n\
            Reconstruted Proposal: {}'.format(
        ' x '.join(list(map(str, out.shape))),
    ))
