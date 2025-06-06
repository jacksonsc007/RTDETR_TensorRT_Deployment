# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibration utilities."""

import math
import types
import warnings
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.utils.distributed import ParallelState

from .conversion import set_quantizer_by_cfg_context
from .nn import SequentialQuantizer, TensorQuantizer
from .utils import (
    get_parallel_state,
    is_quantized_column_parallel_linear,
    is_quantized_layer_with_weight,
    is_quantized_linear,
    is_quantized_row_parallel_linear,
)

__all__ = ["max_calibrate", "awq", "smoothquant", "svdquant"]


@torch.no_grad()
def max_calibrate(
    model: nn.Module, forward_loop: Optional[ForwardLoop] = None, distributed_sync=True
):
    """Calibrate the model using max."""
    enable_stats_collection(model)
    if forward_loop is None:
        # Lets do a weight only calibration
        def forward_loop(model: nn.Module):
            seen_modules = set()
            for name, module in model.named_modules():
                if module in seen_modules:
                    continue
                if is_quantized_layer_with_weight(module) and hasattr(module, "weight_quantizer"):
                    module.weight_quantizer(module.weight)
                    seen_modules.add(module)

    forward_loop(model)
    # NOTE: Ink modified!
    # finish_stats_collection(model)
    finish_stats_collection(model)

    if not distributed_sync:
        return



def enable_stats_collection(model: nn.Module):
    """Enable stats collection for all quantizers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def finish_stats_collection(model: nn.Module, method: Optional[str] = None):
    """Finish stats collection for all quantizers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            from modelopt.torch.quantization.calib import HistogramCalibrator, MaxCalibrator 
            if module._calibrator is not None and not module._dynamic:
                # NOTE: Ink modified. We identify the type of calibrators and apply corresponding logic
                if isinstance(module._calibrator, HistogramCalibrator) and method in ["mse", "entropy", "percentile"]:
                    if module._calibrator.compute_amax(method) is not None:
                        module.load_calib_amax(method)
                    else:
                        module.load_calib_amax(method, strict=False)
                else:
                    calib_amax = module._calibrator.compute_amax()
                    if  calib_amax is not None:
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(strict=False)

                        
            if module.bias_calibrator is not None and module.bias_type == "static":
                module.load_calib_bias()

            module.enable_quant()
            module.disable_calib()


@torch.no_grad()
def disable_pre_quant_scale_and_resmooth(linear: nn.Module, delete_pre_quant_scale: bool = False):
    """Disable pre_quant_scale and resmooth the quantized linear weights."""
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer._enable_pre_quant_scale, (
        "pre_quant_scale should be enabled first!"
    )
    assert hasattr(linear.input_quantizer, "_pre_quant_scale"), (
        "pre_quant_scale should be available"
    )

    pre_quant_scale = linear.input_quantizer._pre_quant_scale.to(torch.float32)

    linear.weight.copy_(
        (linear.weight * pre_quant_scale.squeeze()[None, :]).to(linear.weight.dtype)
    )
    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))

    # Lets not delete the _pre_quant_scale, it might useful later; Instead we will disable it
    linear.input_quantizer._enable_pre_quant_scale = False

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        linear.input_quantizer.amax = linear.input_quantizer._amax_for_smoothing.amax().to(
            linear.weight.dtype
        )

    if delete_pre_quant_scale:
        delattr(linear.input_quantizer, "_pre_quant_scale")
        linear.input_quantizer._enable_pre_quant_scale = False


# A global variable used during AutoQuantize to avoid folding pre_quant_scale to weights
_ENABLE_FOLDING_PQS_TO_WEIGHTS = True


@torch.no_grad()
def _apply_weight_pre_quant_scale(linear, pre_quant_scale):
    if _ENABLE_FOLDING_PQS_TO_WEIGHTS:
        linear.weight.data.copy_(
            (linear.weight * pre_quant_scale.squeeze()[None, :]).to(linear.weight.dtype)
        )
    else:
        linear.weight_quantizer._enable_pre_quant_scale = True
        linear.weight_quantizer.pre_quant_scale = pre_quant_scale.squeeze()[None, :].to(
            linear.weight.dtype
        )

    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))


@torch.no_grad()
def apply_pre_quant_scale_and_smooth(
    linear: nn.Module, pre_quant_scale: Optional[torch.Tensor] = None
):
    """Apply pre_quant_scale and smooth the quantized linear weights.

    If pre_quant_scale is not provided, the existing pre_quant_scale of input_quantizer will be used.
    """
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer.pre_quant_scale is None, "pre_quant_scale should be None first!"

    if pre_quant_scale is None:
        pre_quant_scale = (
            linear.input_quantizer._pre_quant_scale
            if hasattr(linear.input_quantizer, "_pre_quant_scale")
            else None
        )

    assert pre_quant_scale is not None, "pre_quant_scale should be provided or already set"

    assert torch.all(pre_quant_scale > 0), "pre_quant_scale should be positive"

    # pre_quant_scale should be in fp32 for the scaling math to be numerically safe
    pre_quant_scale = pre_quant_scale.to(torch.float32)

    linear.input_quantizer._enable_pre_quant_scale = True
    linear.input_quantizer.pre_quant_scale = pre_quant_scale.to(linear.weight.dtype)

    inv_scale = 1.0 / pre_quant_scale
    _apply_weight_pre_quant_scale(linear, inv_scale)

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        linear.input_quantizer.amax = (
            (linear.input_quantizer._amax_for_smoothing * pre_quant_scale)
            .amax()
            .to(linear.weight.dtype)
        )

        if is_quantized_column_parallel_linear(linear) or is_quantized_row_parallel_linear(linear):
            linear.input_quantizer.sync_amax_across_distributed_group(
                get_parallel_state(linear).tensor_parallel_group
            )


@torch.no_grad()
def smoothquant(model: nn.Module, forward_loop: Optional[ForwardLoop] = None, alpha=1.0):
    """Smooth-Quant variant with per-channel weight scaling.

    The parameters are as described in
    :class:`SmoothQuantCalibConfig <modelopt.torch.quantization.config.SmoothQuantCalibConfig>`.
    """
    # distributed synchornization
    # max_calibrate performs amax sync for data parallel

    # Column parallel:
    # activations:  TPG should have the same pre_quant_scale
    #               This is achieved by syncing act_amax and weight_scale across TPG which is used to
    #               compute pre_quant_scale
    # weights:      no-op

    # Row parallel:
    # activations:  TPG should have same activation amax
    # weights:      TPG should have the same weight amax

    assert forward_loop is not None, "forward_loop must be provided for smoothquant"
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.input_quantizer.is_enabled
            and module.input_quantizer.axis is None
        ):
            module.input_quantizer.axis = -1

    max_calibrate(model, forward_loop)

    smoothed_modules = 0
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            if not hasattr(module.input_quantizer, "_amax"):
                print(f"Warning: {name} is not calibrated, skip smoothing")
                continue
            if module.input_quantizer.num_bits != 8 or module.weight_quantizer.num_bits != 8:
                print(f"Warning: only int8 smoothing is supported, skip {name}")
                continue
            if module.input_quantizer.axis != -1:
                print(f"Warning: only per-channel smoothing is supported, skip {name}")
                continue

            assert module.input_quantizer._amax.numel() > 1, (
                f"Error: {name} has only one channel to smooth"
            )

            # It is important to keep scaling math in fp32 to be numerically safe
            act_amax = module.input_quantizer.amax.float()
            weight_scale = module.weight.abs().amax(dim=0, keepdim=True)

            parallel_group = get_parallel_state(module).tensor_parallel_group
            if is_quantized_column_parallel_linear(module) and parallel_group.is_initialized():
                dist.all_reduce(act_amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
                dist.all_reduce(weight_scale, op=dist.ReduceOp.MAX, group=parallel_group.group)

            scale_a = (weight_scale.pow(1 - alpha) / act_amax.pow(alpha)).squeeze()

            # Now that activation per-channel amax have been collected, use per-tensor quantization for activation
            module.input_quantizer._amax_for_smoothing = act_amax
            module.input_quantizer.reset_amax()
            module.input_quantizer.axis = None
            module.input_quantizer.amax = act_amax.amax()

            # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
            epsilon = 1.0 / (1 << 31)
            if scale_a.min() <= epsilon:
                zero_mask = act_amax <= epsilon
                scale_a[zero_mask] = 1
            scale_a = scale_a.clamp(min=1e-4, max=1e4)
            apply_pre_quant_scale_and_smooth(module, scale_a)

            smoothed_modules += 1
    print(f"Smoothed {smoothed_modules} modules")


def _smoothquant_fasteval(model: nn.Module):
    """Hacky implementation of Smooth-Quant. Copied from monkey-quant."""
    smoothed_modules = 0
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            if not hasattr(module.input_quantizer, "_amax"):
                print(f"Warning: {name} is not calibrated, skip smoothing")
                continue
            if module.input_quantizer.num_bits != 8 or module.weight_quantizer.num_bits != 8:
                print(f"Warning: only int8 smoothing is supported, skip {name}")
                continue
            if module.input_quantizer.axis != -1:
                print(f"Warning: only per-channel smoothing is supported, skip {name}")
                continue

            assert module.input_quantizer._amax.numel() > 1
            delattr(module.weight_quantizer, "_amax")

            # It is important to keep scaling math in fp32 to be numerically safe
            act_amax = module.input_quantizer.amax.float()
            if act_amax.shape[0] == 1:
                act_amax = act_amax.squeeze(0)
            # If model is split across devices, this tensor may be on wrong one
            act_amax = act_amax.to(module.weight.device)

            max_bound = module.input_quantizer.maxbound
            scale_a = max_bound / act_amax
            # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
            epsilon = 1.0 / (1 << 31)
            if act_amax.min() <= epsilon:
                zero_mask = act_amax <= epsilon
                scale_a[zero_mask] = 1
            inv_scale_a = act_amax / max_bound

            module.weight.data.copy_(
                (module.weight_quantizer(inv_scale_a * module.weight.float()) * scale_a).to(
                    module.weight.dtype
                )
            )
            module.weight_quantizer.disable()

            smoothed_modules += 1
    print(f"Smoothed {smoothed_modules} modules")


def awq(
    model: nn.Module,
    algorithm: str = "awq_lite",
    forward_loop: Optional[ForwardLoop] = None,
    **kwargs,
):
    """Apply AWQ to the model."""
    with SequentialQuantizer.replace_sequential_quantizer_with_single_quantizer(model):
        if algorithm in ["awq_full", "awq_lite"]:
            awq_lite(model, forward_loop, **kwargs)

        if algorithm in ["awq_full", "awq_clip"]:
            awq_clip(model, forward_loop, **kwargs)

    for name, module in model.named_modules():
        if is_quantized_linear(module) and isinstance(module.weight_quantizer, SequentialQuantizer):
            max_calibrate(module, lambda linear: linear.weight_quantizer(module.weight))
    # If there are any uncalibrated quantizers, calibrate them using max
    if any(
        not module._disabled and module.amax is None
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer)
    ):
        from .model_quant import calibrate

        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer) and hasattr(module, "_amax"):
                # Already calibrated modules should not be calibrated again
                # This is true for the first weight quantizer of the sequential weight quantizers
                assert module.is_enabled, "calibrated modules should be enabled"
                module.disable()

        calibrate(model, algorithm="max", forward_loop=forward_loop)

        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer) and hasattr(module, "_amax"):
                module.enable()


@torch.no_grad()
def awq_lite(
    model: nn.Module,
    forward_loop: ForwardLoop,
    alpha_step: float = 0.1,
    debug: bool = False,
    **kwargs,
):
    """Lite version of AWQ.

    The parameters are as described in
    :class:`AWQLiteCalibConfig <modelopt.torch.quantization.config.AWQLiteCalibConfig>`.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.
        alpha_step: Step size for the searching awq_lite hyper parameter alpha. alpha is in [0, 1].
        debug: If True, module's search metadata will be kept as a module attribute named `awq_lite`.

    """
    assert forward_loop is not None, "forward_loop must be provided for awq_lite"

    class AWQLiteHelper:
        cache_mode: bool = False

        def __init__(self, module):
            self.act_scale = 0.0
            self.num_cache_steps = 0
            self.num_search_steps = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)
            self.weight_scale = get_weight_scale(module.weight, self.block_size)
            self.loss = {k.item(): 0.0 for k in torch.arange(0, 1.0 + alpha_step, alpha_step)}
            self.best_scale = None
            self.best_alpha = None
            self.is_input_quantized = module.input_quantizer.is_enabled

    def get_weight_scale(weight, block_size=None):
        org_shape = weight.shape
        slice_after_padding = None
        if block_size:
            if org_shape[-1] % block_size != 0:
                slice_after_padding = slice(org_shape[-1])
                weight = F.pad(weight, (0, block_size - org_shape[-1] % block_size), "constant", 0)
                org_shape = weight.shape
            weight = weight.contiguous().view(-1, block_size)
        weight_abs_amax = weight.abs().amax(dim=1, keepdim=True)
        scale = weight.abs() / (weight_abs_amax + torch.finfo(weight.dtype).tiny)
        scale = scale.view(org_shape)
        if slice_after_padding is not None:
            scale = scale[..., slice_after_padding]
        scale = scale.mean(0).to(torch.float32)
        return scale

    def get_act_scale(x):
        return x.abs().contiguous().view(-1, x.shape[-1]).mean(0).to(torch.float32)

    def get_scale(x_max, w_max, alpha, tensor_parallel_group=None, dtype=torch.float32):
        scales = (
            (x_max.pow(alpha) / (w_max.pow(1 - alpha) + torch.finfo(w_max.dtype).tiny))
            .clamp(min=1e-4, max=1e4)
            .view(-1)
        )
        scales = (scales / (scales.max() * scales.min()).sqrt()).view(-1)
        if tensor_parallel_group and tensor_parallel_group.is_initialized():
            dist.all_reduce(scales, op=dist.ReduceOp.SUM, group=tensor_parallel_group.group)
            scales /= tensor_parallel_group.world_size()
        return scales.to(dtype)

    def update_loss(self, out, out_actual, alpha):
        out_actual = out_actual[0] if isinstance(out_actual, tuple) else out_actual
        out = out[0] if isinstance(out, tuple) else out
        loss = (out - out_actual).float().pow(2).mean().item()
        self.awq_lite.loss[alpha] += loss

    def update_best_params(self):
        self.awq_lite.best_alpha = min(self.awq_lite.loss, key=self.awq_lite.loss.get)
        self.awq_lite.best_scale = get_scale(
            self.awq_lite.act_scale,
            self.awq_lite.weight_scale,
            self.awq_lite.best_alpha,
            (
                get_parallel_state(self).tensor_parallel_group
                if is_quantized_column_parallel_linear(self)
                else None
            ),
            self.weight.dtype,
        )

    def forward(self, input, *args, **kwargs):
        # Collect actual output without quantization
        self.weight_quantizer.disable()
        if hasattr(self.input_quantizer, "_pre_quant_scale"):
            delattr(self.input_quantizer, "_pre_quant_scale")
        if hasattr(self.weight_quantizer, "_pre_quant_scale"):
            delattr(self.weight_quantizer, "_pre_quant_scale")
        out_actual = self._forward_no_awq(input, *args, **kwargs)
        self.weight_quantizer.enable()

        if AWQLiteHelper.cache_mode:
            # NOTE: Input can be on different device than weights
            self.awq_lite.act_scale += get_act_scale(input).to(self.awq_lite.weight_scale.device)
            self.awq_lite.num_cache_steps += 1
            if self.awq_lite.is_input_quantized:
                with set_quantizer_by_cfg_context(self.input_quantizer, {"*": {"enable": True}}):
                    # TODO: Why? Calling max_calibrate on the linear layer instead of input_quantizer
                    # is causing memory leakage due to the garbage collection not working as expected
                    # max_calibrate(self, lambda linear: linear.input_quantizer(input))
                    max_calibrate(self.input_quantizer, lambda quantizer: quantizer(input))
            return out_actual

        for alpha in self.awq_lite.loss.keys():
            awq_scale = get_scale(
                self.awq_lite.act_scale,
                self.awq_lite.weight_scale,
                alpha,
                (
                    get_parallel_state(self).tensor_parallel_group
                    if is_quantized_column_parallel_linear(self)
                    else None
                ),
                self.weight.dtype,
            )
            self.input_quantizer.pre_quant_scale = 1 / awq_scale
            self.weight_quantizer.pre_quant_scale = awq_scale
            out = self._forward_no_awq(input, *args, **kwargs)

            update_loss(self, out, out_actual, alpha)

        self.awq_lite.num_search_steps += 1

        # Now forward the actual output without any quantization
        return out_actual

    for name, module in model.named_modules():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            module._forward_no_awq = module.forward
            module.awq_lite = AWQLiteHelper(module)
            module.forward = types.MethodType(forward, module)

            if module.input_quantizer.is_enabled:
                module.input_quantizer.disable()
                if module.input_quantizer.axis not in [None, -1]:
                    raise NotImplementedError(
                        "input quantization needs to be per-tensor or None for AWQ algorithm"
                    )
                module.input_quantizer.axis = -1

    # Collect activation scale values
    AWQLiteHelper.cache_mode = True
    print("Caching activation statistics for awq_lite...")
    forward_loop(model)

    # If input_quantizers were enabled they would be calibrated in caching mode
    # Call max_calibrate to sync amax across distributed groups
    max_calibrate(model, lambda model: None)

    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and hasattr(module, "awq_lite")
            and module.awq_lite.num_cache_steps > 0
        ):
            module.awq_lite.act_scale /= module.awq_lite.num_cache_steps

    AWQLiteHelper.cache_mode = False
    print("Searching awq_lite parameters...")
    forward_loop(model)

    for name, module in model.named_modules():
        if hasattr(module, "awq_lite"):
            if module.awq_lite.num_cache_steps > 0:
                assert module.awq_lite.num_search_steps > 0, (
                    "Calling `forward_loop(model)` the second time did not forward data through the"
                    " model. Please provide a valid `forward_loop` function that can be used to"
                    " forward data through the model many times."
                )

                update_best_params(module)
                delattr(module.weight_quantizer, "_pre_quant_scale")
                delattr(module.input_quantizer, "_pre_quant_scale")
                if module.awq_lite.is_input_quantized:
                    assert module.input_quantizer.amax is not None
                    module.input_quantizer.to(module.weight.device)
                    module.input_quantizer._amax_for_smoothing = module.input_quantizer.amax
                    module.input_quantizer.reset_amax()
                    module.input_quantizer.axis = None
                    module.input_quantizer.amax = module.input_quantizer._amax_for_smoothing.amax()
                    module.input_quantizer.enable()

                apply_pre_quant_scale_and_smooth(module, 1.0 / module.awq_lite.best_scale)

            if not debug:
                delattr(module, "awq_lite")

            # Restore forward function; Why is this dangerous?
            module.forward = module._forward_no_awq
            delattr(module, "_forward_no_awq")


@torch.no_grad()
def awq_clip(
    model: nn.Module,
    forward_loop: ForwardLoop,
    max_co_batch_size: int = 1024,
    max_tokens_per_batch: int = 64,
    min_clip_ratio: float = 0.5,
    shrink_step: float = 0.05,
    debug: bool = False,
    **kwargs,
):
    """AWQ-Clip variant.

    The parameters are as described in
    :class:`AWQClipCalibConfig <modelopt.torch.quantization.config.AWQClipCalibConfig>`.

    Args:
        model: Model to calibrate.
        forward_loop: A callable that runs the forward pass of the model.
        max_co_batch_size: Maximum output channel batch size while searching clip values. Reduce this number if CUDA Out
            of Memory error occurs.
        max_tokens_per_batch: Maximum tokens per batch while searching clip values. The total tokens used for
            clip search would be max_tokens_per_batch * number of batches. Original AWQ uses a total of 512 tokens to
            search for clip values.
        min_clip_ratio: Minimum clip ratio to search for. It should be in (0, 1.0). Clip will search for the optimal
            clipping value in the range [original block amax * min_clip_ratio, original block amax].
        shrink_step: Step size to search for clip values. It should be in range (0, 1.0].
        debug: If True, module's search metadata will be kept as a module attribute named `awq_clip`.
    """
    assert forward_loop is not None, "forward_loop must be provided for awq_clip"

    class AWQClipHelper:
        def __init__(self, module):
            self.num_tokens = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)

            # Cache the original amax
            module.weight_quantizer.reset_amax()
            enable_stats_collection(module.weight_quantizer)
            module.weight_quantizer(module.weight)
            finish_stats_collection(module.weight_quantizer)
            self.w_amax = module.weight_quantizer.amax.clone()

            co, ci = module.weight.shape
            clip_ratios = [
                round(float(k), 2) for k in torch.arange(min_clip_ratio, 1.0, shrink_step)
            ] + [1.0]
            if self.is_per_tensor_clip(module):
                self.loss = {k: torch.tensor(0.0, device=module.weight.device) for k in clip_ratios}
            else:
                self.loss = {
                    k: torch.zeros(
                        (co, math.ceil(ci / self.block_size)), device=module.weight.device
                    )
                    for k in clip_ratios
                }
            self.best_clip_val = None
            self.best_loss = None

            self.is_input_quantized = module.input_quantizer.is_enabled

        def is_per_tensor_clip(self, module):
            quantizer = module.weight_quantizer
            is_dynamic_w_per_tensor = (
                hasattr(quantizer, "block_sizes")
                and quantizer.block_sizes.get("type", None) == "dynamic"
                and quantizer.axis is None
            )
            is_per_tensor = quantizer.axis is None and quantizer.block_sizes is None
            return is_dynamic_w_per_tensor or is_per_tensor

    def update_best_params(self):
        self.awq_clip.best_loss = torch.ones_like(self.awq_clip.w_amax) * float("inf")
        self.awq_clip.best_clip_val = torch.zeros_like(self.awq_clip.w_amax)

        for shrink, loss in self.awq_clip.loss.items():
            loss = loss.view_as(self.awq_clip.w_amax)
            indices = loss < self.awq_clip.best_loss
            self.awq_clip.best_loss = torch.where(indices, loss, self.awq_clip.best_loss)
            self.awq_clip.best_clip_val = torch.where(
                indices, self.awq_clip.w_amax * shrink, self.awq_clip.best_clip_val
            )

    def _clip_search(self, inputs, co_bsz=256, max_tokens=16):
        weight = self.weight
        self.weight_quantizer.enable()

        if self.awq_clip.is_per_tensor_clip(self):
            # In NVFP4, only the per-tensor amax is clipped
            out_actual = inputs @ self.weight.T
            original_amax = self.weight_quantizer.amax.clone()
            self.awq_clip.num_tokens += inputs.shape[0]
            for shrink in self.awq_clip.loss.keys():
                self.weight_quantizer.amax = original_amax * shrink
                out = inputs @ self.weight_quantizer(self.weight).T
                loss = (out - out_actual).float().pow(2).mean()
                self.awq_clip.loss[shrink] += loss
        else:
            # weight  [co, ci] -> [co, 1, n_block, block_size]
            # inputs  [..., ci] -> [1, max_tokens, n_block, block_size]

            inputs = inputs.view(-1, inputs.shape[-1])  # _, ci
            # Select max_tokens from the total input tokens of count batch * n_token
            inputs = inputs[0 :: max(1, inputs.shape[0] // max_tokens)]  # max_tokens, ci
            self.awq_clip.num_tokens += inputs.shape[0]

            block_size = self.awq_clip.block_size
            co, ci = weight.shape
            if ci % block_size != 0:
                weight = F.pad(weight, (0, block_size - ci % block_size), "constant", 0)
                inputs = F.pad(inputs, (0, block_size - ci % block_size), "constant", 0)
                ci = weight.shape[-1]

            weight = weight.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size

            # 1, max_tokens, n_block, block_size
            inputs = inputs.reshape(1, inputs.shape[0], -1, block_size)

            for co_batch in range(math.ceil(co / co_bsz)):
                w = weight[co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)]

                org_out = (inputs * w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                for shrink in self.awq_clip.loss.keys():
                    self.weight_quantizer.amax = self.awq_clip.w_amax * shrink
                    quantized_clipped_weight = self.weight_quantizer(self.weight)
                    cur_w = quantized_clipped_weight[
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ]
                    if cur_w.shape[-1] % block_size != 0:
                        cur_w = F.pad(
                            cur_w, (0, block_size - cur_w.shape[-1] % block_size), "constant", 0
                        )
                    cur_w = cur_w.reshape(w.shape)
                    cur_out = (inputs * cur_w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                    # co_bsz, n_block
                    loss = (cur_out - org_out).float().pow(2).mean(dim=1)

                    parallel_group = get_parallel_state(self).data_parallel_group
                    if parallel_group.is_initialized():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=parallel_group.group)
                        loss /= parallel_group.world_size()

                    del cur_out, cur_w
                    self.awq_clip.loss[shrink][
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ] += loss
                del org_out

    def forward(name, self, input, *args, **kwargs):
        # input shape : (..., cin)
        # weight shape : (cout, cin)
        # Disable quantization
        self.weight_quantizer.disable()
        if self.awq_clip.is_input_quantized:
            self.input_quantizer.enable()
            max_calibrate(self.input_quantizer, lambda input_quantizer: input_quantizer(input))
            self.input_quantizer.disable()
        try:
            _clip_search(self, self.input_quantizer(input), max_co_batch_size, max_tokens_per_batch)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(
                    f"Clip search on {name} failed due to CUDA out of memory, try reducing"
                    " max_co_batch_size"
                ) from e
            raise RuntimeError(e)

        # Disable quantization
        self.weight_quantizer.disable()
        return self._forward_no_awq(input, *args, **kwargs)

    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.weight_quantizer.is_enabled
            and module.weight_quantizer.block_sizes is not None
        ):
            module._forward_no_awq = module.forward
            module.forward = types.MethodType(partial(forward, name), module)
            module.awq_clip = AWQClipHelper(module)

    print("Estimating awq_clip parameters...")
    forward_loop(model)

    for name, module in model.named_modules():
        if is_quantized_linear(module) and hasattr(module, "awq_clip"):
            if module.awq_clip.num_tokens > 0:
                update_best_params(module)

                # Load the best clip value (amax)
                module.weight_quantizer.amax = module.awq_clip.best_clip_val
                module.weight_quantizer.enable()
                if module.awq_clip.is_input_quantized:
                    module.input_quantizer.enable()

            if not debug:
                delattr(module, "awq_clip")

            # Restore forward function
            module.forward = module._forward_no_awq
            delattr(module, "_forward_no_awq")


def _get_awq_quantizer_block_size(tensor: torch.Tensor, quantizer: TensorQuantizer):
    if quantizer.block_sizes is None:
        return None
    if -1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[-1]
    elif 1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[1]
    else:
        raise ValueError("AWQ requires block quantization along -1 axis")
    return blocksize


def svdquant(
    model: nn.Module,
    forward_loop: Optional[ForwardLoop] = None,
    lowrank: int = 32,
    **kwargs,
):
    """Lite version of SVDQuant.

    The parameters are as described in
    :class:`SVDQuantConfig <modelopt.torch.quantization.config.SVDQuantConfig>`.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    """
    # AWQ-Lite initially calibrates only linear quantizers, skipping FMHA quantizers.
    # Enable calibration for BMM quantizers here, so AWQ can perform max calibration for BMMs.
    # Otherwise, AWQ-Lite will require an additional calibration round to activate FMHA quantizers.
    enable_stats_collection(model)
    awq(model, "awq_lite", forward_loop, **kwargs)

    for name, module in model.named_modules():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            print(f"SVD {name}")
            u, s, vt = torch.linalg.svd(module.weight.data.double())
            if u.shape[1] < lowrank or vt.shape[0] < lowrank:
                warnings.warn(
                    "The low-rank dimensions do not match the layer dimensions. "
                    "Please verify your configuration and model settings. "
                    "SVD will be skipped for this layer {name}."
                )
                continue
            us = u[:, :lowrank] * s[:lowrank]
            vt = vt[:lowrank]
            device = module.weight.device
            dtype = module.weight.dtype
            module.weight_quantizer.svdquant_lora_a = vt.to(device).to(dtype)
            module.weight_quantizer.svdquant_lora_b = us.to(device).to(dtype)
            module.weight.data.sub_(
                module.weight_quantizer.svdquant_lora_b @ module.weight_quantizer.svdquant_lora_a
            )
            module.weight_quantizer.reset_amax()
            max_calibrate(
                module,
                lambda module: module.weight_quantizer(module.weight),
            )
