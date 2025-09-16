# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from packaging.version import Version

from sglang.srt.distributed import get_device_group
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
    supports_custom_op,
    is_sm90_supported,
    is_sm100_supported,
    is_blackwell
)


_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

if _is_cuda:
    if _is_flashinfer_available:
        from flashinfer.norm import fused_add_rmsnorm
    else:
        from sgl_kernel import fused_add_rmsnorm
    from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm, rmsnorm

if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    import vllm
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

    _vllm_version = Version(vllm.__version__)

logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu


# 添加对称内存相关的全局变量
_global_symm_mem_hdl = None
_global_staging_buffer = None
_global_residual_symm_mem_hdl = None
_global_residual_buffer = None
_global_max_ctas = None
_symm_mem_initialized = False

def initialize_global_symm_memory(hidden_size):
    """
    全局初始化对称内存，只执行一次
    """
    global _global_symm_mem_hdl, _global_staging_buffer, _global_residual_buffer, _global_residual_symm_mem_hdl, _global_max_ctas, _symm_mem_initialized
    
    # 如果已经初始化且不强制重新初始化，直接返回
    if _symm_mem_initialized:
        return _global_symm_mem_hdl, _global_staging_buffer, _global_residual_buffer, _global_max_ctas
    
    # 检查是否在分布式环境中
    if not dist.is_initialized():
        logger.warning("Distributed environment not initialized, skipping symm_mem setup")
        return None, None, None
    
    # 检查是否有多个GPU
    world_size = dist.get_world_size()
    if world_size <= 1:
        logger.info("Single GPU detected, symm_mem not needed")
        return None, None, None
    
    try:
        # 计算缓冲区大小
        CHUNK_SIZE = global_server_args_dict.get("chunked_prefill_size", 2048) + 512        
        logger.info(f"Initializing global symm_mem with CHUNK_SIZE={CHUNK_SIZE}, hidden_size={hidden_size}")
        
        # 创建暂存缓冲区
        _global_staging_buffer = symm_mem.empty(
            (CHUNK_SIZE, hidden_size),
            device="cuda",
            dtype=torch.bfloat16
        )
        
        # 获取设备组
        device_group = get_device_group()
        
        # 建立对称内存句柄
        _global_symm_mem_hdl = symm_mem.rendezvous(
            _global_staging_buffer, 
            device_group
        )
        # 为residual创建独立的对称内存区域
        _global_residual_buffer = symm_mem.empty(
            (CHUNK_SIZE, hidden_size),
            device="cuda", 
            dtype=torch.bfloat16
        )
        _global_residual_symm_mem_hdl = symm_mem.rendezvous(_global_residual_buffer, device_group)
        
        # 设置MAX_CTAS（可以根据GPU架构调整）
        _global_max_ctas = 16
        
        _symm_mem_initialized = True
        
        logger.info("Global symm_mem initialized successfully")
        return _global_symm_mem_hdl, _global_staging_buffer, _global_residual_buffer, _global_max_ctas
        
    except Exception as e:
        logger.error(f"Failed to initialize global symm_mem: {e}")
        _symm_mem_initialized = False
        return None, None, None

def fused_rs_ln_ag_cta(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    symm_mem_hdl, residual_symm_mem_hdl, rank: int, world_size: int, MAX_CTAS: int, offset: int,
    variance_epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:    
    import sgl_kernel
    sgl_kernel.fused_rs_ln_ag_cta(
            x,
            residual,
            weight,
            symm_mem_hdl.multicast_ptr + offset,
            residual_symm_mem_hdl.multicast_ptr + offset,
            symm_mem_hdl.signal_pad_ptrs_dev,
            rank,
            world_size,
            MAX_CTAS,
            variance_epsilon
        )
    # if layer_id == 1:
    #     print("x: ", x.shape, x)
    #     print("residual: ", residual.shape, residual)
    return x, residual

class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            self._forward_method = self.forward_aiter

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            if _vllm_version < Version("0.9"):
                fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
                return x, residual
            else:
                residual_out = torch.empty_like(x)
                output = torch.empty_like(x)
                fused_add_rms_norm(
                    output,
                    x,
                    residual_out,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return output, residual_out
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual)

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """

        if residual is not None:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size
            from sglang.srt.layers.flashinfer_comm_fusion import (
                flashinfer_allreduce_residual_rmsnorm,
            )
            # if is_blackwell():
            if False:
                fused_op = (
                    torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm
                    if supports_custom_op()
                    else flashinfer_allreduce_residual_rmsnorm
                )
                # print("flashinfer_allreduce_residual_rmsnorm")
                if get_tensor_model_parallel_world_size() > 1:
                    fused_result = fused_op(
                        input_tensor=x,
                        residual=residual,
                        weight=self.weight,
                        eps=self.variance_epsilon,
                    )
                    if fused_result[0] is not None:
                        # print("fused_result[0]: ", fused_result[0].shape, fused_result[0])
                        # print("fused_result[1]: ", fused_result[1].shape, fused_result[1])
                        return fused_result
            else:
                # print("fused_rs_ln_ag_cta")
                world_size = get_tensor_model_parallel_world_size()
                rank = dist.get_rank()
                hidden_size = x.shape[1]
                num_token = x.shape[0]
                tokens_per_rank = (num_token + world_size - 1) // world_size  # 向上取整  
                start_idx = rank * tokens_per_rank
                end_idx = (rank + 1) * tokens_per_rank
                # print("start_idx: ", start_idx)
                # print("end_idx: ", end_idx)
                # print("num_token: ", num_token)
                offset = start_idx * x.shape[1] * x.element_size()
                                
                initialize_global_symm_memory(hidden_size)
                # 检查staging buffer大小是否足够
                required_tokens = x.shape[0]
                max_tokens = _global_staging_buffer.shape[0]                
                if required_tokens > max_tokens:
                    logger.warning(f"Required tokens {required_tokens} exceed staging buffer capacity {max_tokens}")
                    exit(-1)
                    
                _global_staging_buffer[:num_token].copy_(x)
                # print("residual: ", residual.shape, residual)
                _global_residual_buffer[:num_token].copy_(residual)
                # print("residual_buffer: ", _global_residual_buffer[:num_token].shape, _global_residual_buffer[:num_token])  
                fused_rs_ln_ag_cta(
                    _global_staging_buffer[start_idx : end_idx],
                    _global_residual_buffer[start_idx : end_idx],
                    self.weight.data,
                    _global_symm_mem_hdl,
                    _global_residual_symm_mem_hdl,
                    rank,
                    world_size,
                    16,
                    offset,
                    self.variance_epsilon
                )
                # 关键：必须同步residual，因为它被kernel修改了
                # device_group = get_device_group()
                # dist.all_reduce(_global_residual_buffer[:num_token], group=device_group)
                # torch.cuda.synchronize()
                # print("_global_staging_buffer: ", _global_staging_buffer[:num_token].shape, _global_staging_buffer[:num_token])
                # print("_global_residual_buffer: ", _global_residual_buffer[:num_token].shape, _global_residual_buffer[:num_token])
                return _global_staging_buffer[:num_token], _global_residual_buffer[:num_token]
        return self.forward(x, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        # Re-dispatch
        if _is_hip:
            self._forward_method = self.forward_native

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x if residual is None else (x, residual)


class Gemma3RMSNorm(CustomOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


if not (
    _is_cuda or _is_hip or _is_npu or (_is_cpu and _is_cpu_amx_available) or _is_xpu
):
    logger.info(
        "sgl-kernel layernorm implementation is not available on current platform. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
