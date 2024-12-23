from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange


# 使用 namedtuple 定义一个高效注意力机制的配置结构体
"""
Config 结构体包含以下字段：
- enable_flash (bool): 是否启用 Flash 注意力机制
- enable_math (bool): 是否启用数学运算注意力机制
- enable_mem_efficient (bool): 是否启用内存高效注意力机制
"""
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    Args:
        val: 要检查的值。

    Returns:
        bool: 如果值存在（不为 None）则返回 True，否则返回 False。
    """
    return val is not None


def once(fn):
    """
    装饰器函数，用于确保被装饰的函数只执行一次。

    Args:
        fn (callable): 需要被装饰的函数。

    Returns:
        callable: 装饰后的函数，只执行一次。
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 使用 once 装饰器创建一个只打印一次的 print 函数
print_once = once(print)



class Attend(nn.Module):
    """
    Attend 模块实现了自注意力机制，支持因果掩码和 Flash Attention。

    Args:
        dropout (float, optional): Dropout 概率，默认为 0。
        causal (bool, optional): 是否使用因果掩码，默认为 False。
        use_flash (bool, optional): 是否使用 Flash Attention，默认为 False。
    """
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash = False
    ):
        super().__init__()
        # 定义注意力 Dropout 层
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        # 注册一个缓冲区用于存储掩码，不持久化到模型状态中
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        # 确定在 CUDA 和 CPU 上的高效注意力配置
        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # A100 GPU 使用 Flash Attention
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            # 仅使用 Flash Attention
            self.cuda_config = Config(True, False, False)
        else:
            # 非 A100 GPU 使用数学或内存高效注意力
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            # 使用数学计算和内存高效注意力
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        """
        生成因果掩码。如果已经存在掩码且其大小足够，则复用；否则，生成新的掩码。

        Args:
            n (int): 序列长度
            device (torch.device): 张量所在的设备

        Returns:
            torch.Tensor: 因果掩码
        """
        if exists(self.mask) and self.mask.shape[-1] >= n:
            # 如果已有掩码且足够大，则复用
            return self.mask[:n, :n]

        # 生成上三角布尔掩码
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        # 注册新的掩码到缓冲区
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        """
        使用 Flash Attention 计算注意力机制。

        Args:
            q (torch.Tensor): 查询张量，形状为 (batch_size, heads, q_len, head_dim)
            k (torch.Tensor): 键张量，形状为 (batch_size, heads, k_len, head_dim)
            v (torch.Tensor): 值张量，形状为 (batch_size, heads, k_len, head_dim)
            mask (Optional[torch.Tensor]): 注意力掩码，形状为 (batch_size, q_len)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, heads, q_len, head_dim)
        """
        # 获取张量形状信息
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])
        # 推荐用于多查询单键值注意力，由 Tri Dao 提出
        # kv 张量形状 torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            # 如果键的维度为 3，则扩展到与查询相同的形状
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            # 如果值的维度为 3，则扩展到与查询相同的形状
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        # 检查掩码是否存在并扩展到兼容的形状
        # 掩码的形状为 B L，因此需要扩展到 B H N L

        if exists(mask):
            # 调整形状以匹配注意力计算
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # 检查是否有兼容的设备用于 Flash Attention

        # 根据设备选择配置
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            # 调用缩放点积注意力
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        """
        前向传播函数，执行注意力机制。

        Args:
            q (torch.Tensor): 查询张量，形状为 (batch_size, heads, seq_len, head_dim)
            k (torch.Tensor): 键张量，形状为 (batch_size, heads, seq_len, head_dim) 或 (batch_size, seq_len, head_dim)
            v (torch.Tensor): 值张量，形状为 (batch_size, heads, seq_len, head_dim) 或 (batch_size, seq_len, head_dim)
            mask (Optional[torch.Tensor]): 注意力掩码，形状为 (batch_size, seq_len)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, heads, seq_len, head_dim)
        """
        # 获取序列长度和设备信息
        n, device = q.shape[-2], q.device
        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            # 如果使用 Flash Attention，则调用 flash_attn 方法
            return self.flash_attn(q, k, v, mask = mask)

        # 根据键的维度确定 einsum 公式
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # similarity
        # 计算相似度

        # 计算缩放点积相似度
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        # 应用键填充掩码

        if exists(mask):
            # 调整掩码形状以匹配相似度张量
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # 应用掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        # 应用因果掩码

        if self.causal:
            # 获取因果掩码
            causal_mask = self.get_mask(n, device)
            # 应用因果掩码
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重

        # 计算 softmax 注意力权重
        attn = sim.softmax(dim=-1)
        # 应用 Dropout
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
