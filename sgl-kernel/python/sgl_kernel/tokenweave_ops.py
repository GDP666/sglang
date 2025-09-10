from typing import Optional
import torch


def rms_norm_inplace(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """RMS Norm in-place kernel"""
    torch.ops.sgl_kernel.rms_norm_inplace.default(result, input, weight, epsilon)


def fused_rs_ln_ag_cta(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
    epsilon: float,
) -> None:
    """Fused ReduceScatter + LayerNorm + AllGather CTA-based kernel"""
    torch.ops.sgl_kernel.fused_rs_ln_ag_cta.default(
        input, residual, weight, mcptr, signal_pads, rank, world_size, MAX_CTAS, epsilon
    )


def fused_add_rms_norm_cta(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    MAX_CTAS: int,
    epsilon: float,
) -> None:
    """Fused Add + RMS Norm CTA-based kernel"""
    torch.ops.sgl_kernel.fused_add_rms_norm_cta.default(
        input, residual, weight, MAX_CTAS, epsilon
    )


def fused_rs_ln_cta(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
    epsilon: float,
) -> None:
    """Fused ReduceScatter + LayerNorm CTA-based kernel"""
    torch.ops.sgl_kernel.fused_rs_ln_cta.default(
        input, residual, weight, mcptr, signal_pads, rank, world_size, MAX_CTAS, epsilon
    )


def multimem_ar_cta(
    input: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
) -> None:
    """Multi-memory AllReduce CTA-based kernel"""
    torch.ops.sgl_kernel.multimem_ar_cta.default(
        input, mcptr, signal_pads, rank, world_size, MAX_CTAS
    )


def multimem_rs_cta(
    input: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
) -> None:
    """Multi-memory ReduceScatter CTA-based kernel"""
    torch.ops.sgl_kernel.multimem_rs_cta.default(
        input, mcptr, signal_pads, rank, world_size, MAX_CTAS
    )


def multimem_ag_cta(
    input: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
) -> None:
    """Multi-memory AllGather CTA-based kernel"""
    torch.ops.sgl_kernel.multimem_ag_cta.default(
        input, mcptr, signal_pads, rank, world_size, MAX_CTAS
    )


def simple_fusion_rs_ln_ag_cta(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
    epsilon: float,
) -> None:
    """Simple fusion ReduceScatter + LayerNorm + AllGather CTA-based kernel"""
    torch.ops.sgl_kernel.simple_fusion_rs_ln_ag_cta.default(
        input, residual, weight, mcptr, signal_pads, rank, world_size, MAX_CTAS, epsilon
    )