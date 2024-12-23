import torch
from einops import repeat, rearrange


def average_over_durations(values, durs):
    """
        - in:
            - values: B, 1, T_de
            - durs: B, T_en
        - out:
            - avg: B, 1, T_en
    """
    """
    根据持续时间信息对输入的 values 进行平均处理。

    Args:
        values (torch.Tensor): 输入张量，形状为 [B, 1, T_de]
            - B: 批次大小
            - T_de: 目标序列长度
        durs (torch.Tensor): 持续时间张量，形状为 [B, T_en]
            - T_en: 源序列长度

    Returns:
        avg (torch.Tensor): 平均后的张量，形状为 [B, 1, T_en]

    处理步骤:
        1. 计算持续时间的累积和，得到每个持续时间的起始和结束位置。
        2. 对 values 进行累积求和和非零元素的累积计数。
        3. 使用累积和和累积计数计算每个目标时间步的平均值。
    """
    # 计算持续时间的累积和，得到每个持续时间的结束位置
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    # 计算持续时间的累积和，并进行填充，得到每个持续时间的起始位置
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))

    # 对 values 进行累积求和，并进行填充
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    # 对 values 中非零元素进行累积计数，并进行填充
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    # 获取批次大小和目标序列长度
    bs, l = durs_cums_ends.size()
    # 获取表单数量
    n_formants = values.size(1)

    # 重复 durs_cums_starts 以匹配表单数量
    dcs = repeat(durs_cums_starts, 'bs l -> bs n l', n=n_formants)
    # 重复 durs_cums_ends 以匹配表单数量
    dce = repeat(durs_cums_ends, 'bs l -> bs n l', n=n_formants)

    # 计算每个持续时间内的 values 累积和
    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).to(values.dtype)
    # 计算每个持续时间内的非零元素数量
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).to(values.dtype)

    # 计算平均值。如果非零元素数量为 0，则平均值设为 0；否则，计算累积和除以非零元素数量
    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems).to(values.dtype)
    return avg


def create_mask(sequence_length, max_len):
    """
    根据序列长度创建一个掩码。

    Args:
        sequence_length (torch.Tensor): 序列长度张量，形状为 [B]，其中 B 是批次大小。
        max_len (int): 最大序列长度。

    Returns:
        mask (torch.Tensor): 掩码张量，形状为 [B, T]，其中 T 是 max_len。
            掩码中值为 True 的位置表示有效，False 表示无效。

    处理步骤:
        1. 获取输入张量的数据类型和设备。
        2. 创建一个范围张量，从 0 到 max_len - 1。
        3. 重塑序列长度张量和范围张量，以便进行广播比较。
        4. 比较范围张量和序列长度张量，生成掩码。
    """
    # 获取输入张量的数据类型和设备
    dtype, device = sequence_length.dtype, sequence_length.device

    # 创建一个范围张量，从 0 到 max_len - 1
    seq_range = torch.arange(max_len, dtype=dtype, device=device)

    # 重塑序列长度张量，从 [B] 变为 [B, 1]
    sequence_length = rearrange(sequence_length, 'b -> b 1')

    # 重塑范围张量，从 [T] 变为 [1, T]
    seq_range = rearrange(seq_range, 't -> 1 t')

    # 比较范围张量和序列长度张量，生成掩码
    # 掩码中值为 True 的位置表示序列长度大于或等于当前索引
    return seq_range < sequence_length
