import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import math
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from PIL import Image
import requests
import zipfile
import io
from einops import rearrange
import glob

# 필요한 패키지 설치
def install_packages():
    packages = [
        "torch",
        "torchvision",
        "einops",
        "matplotlib",
        "tqdm",
        "requests",
        "pillow",
        "timm"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # timm 패키지 설치 확인
    try:
        import timm
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])

# 패키지 설치 실행
# install_packages()

# timm에서 필요한 함수들 임포트
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 디바이스 설정 (MPS 우선)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"사용 중인 디바이스: {device}")

# FFT 기반 다중 헤드 어텐션 모듈 구현
class FFTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]

        # 어텐션 대신 FFT 적용
        # 2D FFT 형태로 재구성
        h = w = int(math.sqrt(N))
        v_2d = rearrange(v, 'b h (p1 p2) c -> b h p1 p2 c', p1=h, p2=w)
        
        # 실수부와 허수부 분리를 위해 복소수 FFT 수행
        v_fft = torch.fft.fft2(v_2d.float(), dim=(2, 3))
        
        # 복소수 결과를 실수부와 허수부로 분리하여 처리
        v_fft_real = v_fft.real
        v_fft_imag = v_fft.imag
        
        # 실수부와 허수부를 합쳐서 채널 차원에 추가
        v_fft_combined = torch.cat([v_fft_real, v_fft_imag], dim=-1)
        
        # 원래 형태로 변환
        v_transformed = rearrange(v_fft_combined, 'b h p1 p2 c -> b h (p1 p2) c')
        
        # 원래 차원과 맞추기 위한 선형 변환
        v_transformed = v_transformed.reshape(B, self.num_heads, N, -1)
        v_transformed = v_transformed[..., :C//self.num_heads]  # 채널 차원 맞추기
        
        x = rearrange(v_transformed, 'b h n c -> b n (h c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# FFT 기반 윈도우 어텐션 구현
class FFTWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # FFT 처리를 위한 QKV 프로젝션
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, C//num_heads]

        # 윈도우 크기에 맞게 재구성
        H, W = self.window_size
        v_window = rearrange(v, 'b h (p1 p2) c -> b h p1 p2 c', p1=H, p2=W)
        
        # 2D FFT 적용
        v_fft = torch.fft.fft2(v_window.float(), dim=(2, 3))
        
        # 복소수 결과를 실수부와 허수부로 분리하여 처리
        v_fft_real = v_fft.real
        v_fft_imag = v_fft.imag
        
        # 실수부와 허수부를 합쳐서 채널 차원에 추가
        v_fft_combined = torch.cat([v_fft_real, v_fft_imag], dim=-1)
        
        # 원래 형태로 변환
        v_transformed = rearrange(v_fft_combined, 'b h p1 p2 c -> b h (p1 p2) c')
        
        # 원래 차원과 맞추기 위한 처리
        v_transformed = v_transformed.reshape(B_, self.num_heads, N, -1)
        v_transformed = v_transformed[..., :C//self.num_heads]  # 채널 차원 맞추기
        
        # 최종 출력
        x = rearrange(v_transformed, 'b h n c -> b n (h c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Mlp 모듈 정의
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

# 윈도우 파티셔닝 함수
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# 윈도우 역변환 함수
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# 변형된 SWIN Transformer 블록
class FFTSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # 기존 WindowAttention 대신 FFTWindowAttention 사용
        self.attn = FFTWindowAttention(
            dim=dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 윈도우 파티셔닝용 패딩
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 순환 이동 (cyclic shift)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # 윈도우 파티셔닝
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # FFT 기반 윈도우 어텐션 적용
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # 윈도우 병합
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # 역방향 순환 이동 (reverse cyclic shift)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# FFT-SWIN Transformer 계층 구현
class FFTSwinTransformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 블록 빌드
        self.blocks = nn.ModuleList([
            FFTSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 다운샘플링 레이어
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, mask_matrix=None):
        for blk in self.blocks:
            x = blk(x, mask_matrix)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

# PatchEmbed 모듈 정의
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # 패딩
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

# PatchMerging 모듈 정의
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# 세그멘테이션용 FFT-SWIN Transformer 모델
# 세그멘테이션용 FFT-SWIN Transformer 모델
class FFTSwinTransformerSegmentation(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, num_classes=150,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # 패치 임베딩 레이어
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # 위치 인코딩
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 드롭 패스 레이트 설정
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # 인코더 계층 빌드
        self.layers = nn.ModuleList()
        self.features = []  # 특징 맵 저장 (모듈리스트가 아닌 일반 리스트로 변경)
        
        for i_layer in range(self.num_layers):
            layer = FFTSwinTransformerLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)
        
        # 디코더 계층 (간단한 FPN 스타일)
        self.decode_layers = nn.ModuleList()
        # 업샘플링 레이어
        self.upsamples = nn.ModuleList()
        # 합치기 전 차원 조정용 프로젝션 레이어
        self.proj_layers = nn.ModuleList()
        
        # 인코더의 각 레이어 출력을 디코딩하는 레이어 생성
        for i in range(self.num_layers-1, 0, -1):
            in_dim = int(embed_dim * 2 ** i)
            out_dim = int(embed_dim * 2 ** (i-1))
            # 특징 맵 크기를 2배로 업샘플링
            upsample = nn.Sequential(
                nn.Linear(in_dim, out_dim * 4),
                nn.GELU(),
                nn.LayerNorm(out_dim * 4)
            )
            self.upsamples.append(upsample)
            
            # 낮은 레이어 특징과 합치기 전 차원 맞추기
            proj = nn.Linear(out_dim, out_dim)
            self.proj_layers.append(proj)
            
            # 합친 후 처리하는 디코더 레이어
            decode_layer = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            )
            self.decode_layers.append(decode_layer)
        
        # 최종 출력 헤드
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        # 인코딩 과정
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        # 인코더의 각 계층에서 특징 저장
        features = []
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:  # 마지막 레이어 제외 특징 저장
                features.append(x)
            x = layer(x)
        
        # 디코딩 과정 (인코더의 역순)
        for i, (upsample, proj, decode) in enumerate(zip(self.upsamples, self.proj_layers, self.decode_layers)):
            # 특징 인덱스 계산 (역순)
            feat_idx = self.num_layers - 2 - i
            
            # 현재 특징 업샘플링
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = upsample(x)  # B, L, C*4
            x = x.view(B, H, W, 4, -1)
            x = x.permute(0, 3, 1, 2, 4).reshape(B, H*2, W*2, -1)
            x = x.view(B, H*2*W*2, -1)
            
            # 저장된 특징 프로젝션
            skip = proj(features[feat_idx])
            
            # 특징 합치기 (채널 방향으로 연결)
            x = torch.cat([x, skip], dim=-1)
            
            # 디코더 레이어 적용
            x = decode(x)
        
        # 최종 정규화 및 특징 출력
        x = self.norm(x)
        return x
    
    def forward(self, x):
        # 이미지 크기 및 배치 크기 저장
        B, C, H, W = x.shape
        
        # 특징 추출
        x = self.forward_features(x)
        
        # 세그멘테이션 헤드 적용
        x = self.head(x)
        
        # 결과를 이미지 형태로 변환 (B, num_classes, H, W)
        x = x.transpose(1, 2).reshape(B, self.num_classes, self.patches_resolution[0], self.patches_resolution[1])
        
        # 원본 이미지 크기로 업샘플링
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

# ADE20K 데이터셋 클래스
class ADE20KSegmentation(Dataset):
    def __init__(self, root, split='training', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # 이미지 및 라벨 경로 리스트 생성
        self.img_dir = os.path.join(root, 'ADEChallengeData2016', 'images', split)
        self.mask_dir = os.path.join(root, 'ADEChallengeData2016', 'annotations', split)
        
        # 이미지 파일 리스트 가져오기
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        # 파일 경로에서 라벨 경로 생성
        self.mask_paths = []
        for img_path in self.img_paths:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace('.jpg', '.png')
            self.mask_paths.append(os.path.join(self.mask_dir, mask_name))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 이미지 및 마스크 로드
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 이미지 로드 및 변환
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 이미지 및 마스크 크기 조정
        if self.transform is not None:
            image = self.transform(image)
            # 마스크는 별도 변환 적용 (리사이즈만)
            mask = transforms.functional.resize(mask, (image.shape[1], image.shape[2]), interpolation=transforms.InterpolationMode.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        
        # ADE20K 마스크는 0이 경계, 1-150이 클래스임
        # 0을 무시하도록 처리
        mask = mask - 1  # 0-149로 변환
        mask[mask < 0] = 255  # 무시할 값을 255로 설정
        
        return {
            'image': image,
            'mask': mask
        }

# ADE20K 데이터셋 다운로드 및 준비 함수
def setup_ade20k_dataset(batch_size=4, img_size=512):
    # 데이터셋 디렉토리 설정
    data_dir = os.path.expanduser("~/data/ade20k")
    os.makedirs(data_dir, exist_ok=True)
    
    # 데이터셋 다운로드 확인 및 다운로드
    ade_path = os.path.join(data_dir, "ADEChallengeData2016")
    if not os.path.exists(ade_path):
        print("ADE20K 데이터셋을 다운로드합니다...")
        # 데이터셋 URL
        ade_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        
        # 다운로드 및 압축 해제
        try:
            response = requests.get(ade_url, stream=True)
            response.raise_for_status()
            
            # 압축 파일 저장 및 압축 해제
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(data_dir)
            
            print("ADE20K 데이터셋 다운로드 및 압축 해제 완료!")
        except Exception as e:
            print(f"다운로드 오류: {e}")
            
            # 대안으로 wget 및 unzip 사용 시도
            try:
                zip_path = os.path.join(data_dir, "ADEChallengeData2016.zip")
                subprocess.check_call(["wget", ade_url, "-O", zip_path])
                subprocess.check_call(["unzip", zip_path, "-d", data_dir])
                os.remove(zip_path)
                print("대안 방법으로 ADE20K 데이터셋 다운로드 및 압축 해제 완료!")
            except Exception as e2:
                print(f"대안 다운로드 방법도 실패: {e2}")
                print("수동으로 데이터셋을 다운로드하고 압축을 해제해주세요.")
                return None, None
    else:
        print("ADE20K 데이터셋이 이미 존재합니다.")
    
    # 데이터 변환 설정
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 생성
    train_dataset = ADE20KSegmentation(root=data_dir, split='training', transform=train_transform)
    val_dataset = ADE20KSegmentation(root=data_dir, split='validation', transform=val_transform)
    
    # 작은 샘플 데이터셋 생성 (개발 및 디버깅용)
    if len(train_dataset) > 100:
        print(f"전체 훈련 데이터셋 크기: {len(train_dataset)}")
        print("훈련 및 검증을 위해 더 작은 샘플 데이터셋을 사용합니다.")
        
        # 작은 샘플 데이터셋 인덱스 생성
        train_indices = list(range(100))
        val_indices = list(range(50))
        
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 손실 함수 정의
class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=150, ignore_index=255):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, pred, target):
        return self.ce_loss(pred, target)

# 평가 지표 함수
def calculate_metrics(pred, target, num_classes=150, ignore_index=255):
    pred = pred.argmax(dim=1)
    
    # 유효한 픽셀만 고려
    valid_mask = (target != ignore_index)
    
    # IoU 계산
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union > 0:
            iou = intersection / union
            iou_list.append(iou.item())
    
    miou = torch.tensor(iou_list).mean().item() if iou_list else 0.0
    
    # 픽셀 정확도
    acc = (pred[valid_mask] == target[valid_mask]).float().mean().item() if valid_mask.sum() > 0 else 0.0
    
    return {'mIoU': miou, 'Acc': acc}

# 학습 함수
def train(model, train_loader, val_loader, device, num_epochs=50, save_dir="models"):
    # 모델 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 최적화 설정
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 손실 함수
    criterion = SegmentationLoss(num_classes=150, ignore_index=255)
    
    # 로깅 설정
    log_interval = 10
    best_miou = 0.0
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        # 진행 상황 표시
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for i, data in enumerate(train_bar):
            # 데이터 준비
            images = data['image'].to(device)
            masks = data['mask'].to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            
            # 손실 계산
            loss = criterion(outputs, masks)
            
            # 역전파
            loss.backward()
            
            # 가중치 업데이트
            optimizer.step()
            
            # 손실 누적
            train_loss += loss.item()
            
            # 진행 상황 업데이트
            train_bar.set_postfix(loss=loss.item())
            
            # 메모리 절약을 위한 캐시 정리
            if i % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 에폭당 평균 손실
        epoch_train_loss = train_loss / len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0.0
        metrics = {'mIoU': 0.0, 'Acc': 0.0}
        
        # 진행 상황 표시
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for i, data in enumerate(val_bar):
                # 데이터 준비
                images = data['image'].to(device)
                masks = data['mask'].to(device)
                
                # 추론
                outputs = model(images)
                
                # 손실 계산
                loss = criterion(outputs, masks)
                
                # 손실 누적
                val_loss += loss.item()
                
                # 지표 계산
                batch_metrics = calculate_metrics(outputs, masks, num_classes=150)
                metrics['mIoU'] += batch_metrics['mIoU']
                metrics['Acc'] += batch_metrics['Acc']
                
                # 진행 상황 업데이트
                val_bar.set_postfix(loss=loss.item(), mIoU=batch_metrics['mIoU'])
        
        # 에폭당 평균 지표
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_miou = metrics['mIoU'] / len(val_loader)
        epoch_val_acc = metrics['Acc'] / len(val_loader)
        
        # 학습률 업데이트
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, mIoU: {epoch_val_miou:.4f}, Acc: {epoch_val_acc:.4f}")
        print(f"학습률: {current_lr:.6f}")
        
        # 최고 성능 모델 저장
        if epoch_val_miou > best_miou:
            best_miou = epoch_val_miou
            model_path = os.path.join(save_dir, "fft_swin_seg_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'miou': epoch_val_miou,
            }, model_path)
            print(f"새로운 최고 mIoU: {best_miou:.4f}, 모델 저장 완료: {model_path}")
        
        # 정기 체크포인트 저장
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(save_dir, f"fft_swin_seg_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'miou': epoch_val_miou,
            }, model_path)
            print(f"체크포인트 저장 완료: {model_path}")
    
    # 최종 모델 저장
    model_path = os.path.join(save_dir, "fft_swin_seg_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_val_loss,
        'miou': epoch_val_miou,
    }, model_path)
    print(f"최종 모델 저장 완료: {model_path}")
    
    return model, best_miou

# 테스트 함수
def test(model, test_loader, device):
    # 테스트 모드
    model.eval()
    metrics = {'mIoU': 0.0, 'Acc': 0.0}
    
    # 진행 상황 표시
    test_bar = tqdm(test_loader, desc="테스트 중...")
    
    with torch.no_grad():
        for i, data in enumerate(test_bar):
            # 데이터 준비
            images = data['image'].to(device)
            masks = data['mask'].to(device)
            
            # 추론
            outputs = model(images)
            
            # 지표 계산
            batch_metrics = calculate_metrics(outputs, masks, num_classes=150)
            metrics['mIoU'] += batch_metrics['mIoU']
            metrics['Acc'] += batch_metrics['Acc']
            
            # 진행 상황 업데이트
            test_bar.set_postfix(mIoU=batch_metrics['mIoU'], Acc=batch_metrics['Acc'])
    
    # 평균 지표
    avg_miou = metrics['mIoU'] / len(test_loader)
    avg_acc = metrics['Acc'] / len(test_loader)
    
    print(f"테스트 결과:")
    print(f"평균 mIoU: {avg_miou:.4f}")
    print(f"평균 정확도: {avg_acc:.4f}")
    
    return avg_miou, avg_acc

# 세그멘테이션 결과 시각화 함수
def visualize_segmentation(model, image_path, device, output_path=None):
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지 변환
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 추론
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # 결과 처리
    pred = output.argmax(dim=1)[0].cpu().numpy()
    
    # ADE20K 색상 팔레트 생성 (무작위 색상)
    np.random.seed(42)  # 일관된 색상을 위한 시드 설정
    palette = np.random.randint(0, 255, size=(150, 3), dtype=np.uint8)
    # 배경 (무시) 클래스를 검은색으로 설정
    palette[0] = [0, 0, 0]  
    
    # 색상 매핑
    colored_pred = palette[pred]
    
    # 원본 이미지 복원 (정규화 해제)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    orig_img = std * orig_img + mean
    orig_img = np.clip(orig_img, 0, 1)
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title('원본 이미지')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(colored_pred)
    plt.title('세그멘테이션 결과')
    plt.axis('off')
    
    # 결과 오버레이
    plt.subplot(1, 3, 3)
    overlay = orig_img * 0.7 + colored_pred.astype(float) / 255 * 0.3
    plt.imshow(overlay)
    plt.title('오버레이 결과')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"세그멘테이션 결과가 {output_path}에 저장되었습니다.")
    
    plt.show()
    
    return colored_pred

# 메인 함수
def main():
    # 인자 설정
    img_size = 512
    patch_size = 4
    in_chans = 3
    num_classes = 150  # ADE20K 클래스 수
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7
    batch_size = 4
    num_epochs = 30  # 필요에 따라 조정
    
    # 디바이스 설정
    device = get_device()
    print(f"디바이스: {device}")
    
    # 데이터 로더 설정
    train_loader, val_loader = setup_ade20k_dataset(batch_size=batch_size, img_size=img_size)
    
    if train_loader is None or val_loader is None:
        print("데이터 로더를 설정할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    # 모델 생성
    model = FFTSwinTransformerSegmentation(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size
    ).to(device)
    
    # 모델 정보 출력
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 학습
    print("모델 학습을 시작합니다...")
    model, best_miou = train(model, train_loader, val_loader, device, num_epochs=num_epochs)
    
    # 테스트
    print("모델을 테스트합니다...")
    avg_miou, avg_acc = test(model, val_loader, device)
    
    # 샘플 이미지로 세그멘테이션 결과 시각화
    data_dir = os.path.expanduser("~/data/ade20k")
    sample_dir = os.path.join(data_dir, "ADEChallengeData2016", "images", "validation")
    sample_images = glob.glob(os.path.join(sample_dir, "*.jpg"))
    
    if sample_images:
        # 첫 번째 샘플 이미지 사용
        sample_image_path = sample_images[0]
        output_path = "segmentation_result.png"
        
        print(f"샘플 이미지로 세그멘테이션 결과를 시각화합니다: {sample_image_path}")
        visualize_segmentation(model, sample_image_path, device, output_path)
    else:
        print("샘플 이미지를 찾을 수 없어 시각화를 건너뜁니다.")

# 스크립트를 직접 실행할 때 메인 함수 호출
if __name__ == "__main__":
    main()
