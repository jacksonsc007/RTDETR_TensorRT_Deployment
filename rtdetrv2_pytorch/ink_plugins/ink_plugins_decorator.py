import tensorrt as trt
import tensorrt.plugin as trtp
import numpy as np
import numpy.typing as npt
from typing import Tuple
import torch 

namespace = "ink_plugins"
plugin_name = "fused_attn_offset_prediction" # operation name
# Generate the output signature
# Input: int8. [1 ,300, 256]
# Output: Float [1, 300, 288]. It is a concated tensor of [1, 300, 192] (for sampling offsets) and [1, 300, 96] (for attention weights)
@trtp.register(f"{namespace}::{plugin_name}")
def fused_attn_offset_prediction_desc(
    # inputs
    inp0: trtp.TensorDesc,
    # attributes
    input_scale: npt.NDArray[np.float32],
    input_offset: npt.NDArray[np.int8],
    sp_weight_scale: npt.NDArray[np.float32],
    sp_weight_offset: npt.NDArray[np.int8],
    attn_weight_scale: npt.NDArray[np.float32],
    attn_weight_offset: npt.NDArray[np.float32],
    int8_sp_weight: npt.NDArray[np.int8],
    int8_attn_weight: npt.NDArray[np.int8],
    sp_fp32_bias: npt.NDArray[np.float32],
    attn_fp32_bias: npt.NDArray[np.float32],
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    out_desc_1 = inp0.like()
    out_desc_2 = inp0.like()
    out_desc_1.shape_expr[2] = 96 # output for attention weights
    out_desc_2.shape_expr[2] = 192 # output for sampling offsets
    # modify the output from int8 (from int) to fp32
    out_desc_1.dtype = trt.float32
    out_desc_2.dtype = trt.float32
    return out_desc_1, out_desc_2

# compuatation functions
@trtp.impl(f"{namespace}::{plugin_name}")
def fused_attn_offset_prediction_impl(
    # inputs
    inp0: trtp.Tensor,
    # attributes
    input_scale: npt.NDArray[np.float32],
    input_offset: npt.NDArray[np.int8],
    sp_weight_scale: npt.NDArray[np.float32],
    sp_weight_offset: npt.NDArray[np.int8],
    attn_weight_scale: npt.NDArray[np.float32],
    attn_weight_offset: npt.NDArray[np.float32],
    int8_sp_weight: npt.NDArray[np.int8],
    int8_attn_weight: npt.NDArray[np.int8],
    sp_fp32_bias: npt.NDArray[np.float32],
    attn_fp32_bias: npt.NDArray[np.float32],
    # outputs
    outputs: Tuple[ trtp.Tensor ],
    stream: int
) -> None:
    inp_t = torch.as_tensor(inp0, device="cuda") # [1, 300, 256]
    bs, n_queries, n_features = inp_t.shape
    
    # get attributes as tensors
    # Need to reshape since attributes are flattened
    int8_sp_weight_t = torch.as_tensor(int8_sp_weight, device="cuda").reshape(192, 256) # [192, 256]
    int8_attn_weight_t = torch.as_tensor(int8_attn_weight, device="cuda").reshape(96, 256) # [96, 256]
    sp_fp32_bias_t = torch.as_tensor(sp_fp32_bias, device="cuda") # [192]
    attn_fp32_bias_t = torch.as_tensor(attn_fp32_bias, device="cuda") # [96]
    input_scale_t = torch.as_tensor(input_scale, device="cuda") # [1]
    input_offset_t = torch.as_tensor(input_offset, device="cuda") # [1]
    sp_weight_scale_t = torch.as_tensor(sp_weight_scale, device="cuda").reshape(192, 1) # [192]
    sp_weight_offset_t = torch.as_tensor(sp_weight_offset, device="cuda").reshape(192, 1) # [192]
    attn_weight_scale_t = torch.as_tensor(attn_weight_scale, device="cuda").reshape(96, 1) # [96]
    attn_weight_offset_t = torch.as_tensor(attn_weight_offset, device="cuda").reshape(96, 1) # [96]
    
    output_t_attn = torch.as_tensor(outputs[0], device="cuda")
    output_t_sp = torch.as_tensor(outputs[1], device="cuda")
    
    # print(f"\033[96m[DEBUG] int8_sp_weight: {int8_sp_weight.shape}\033[0m")
    # print(f"\033[96m[DEBUG] int8_attn_weight: {int8_attn_weight.shape}\033[0m")
    # compute output of sampling offsets
    # print("\033[96m[INFO] Plugining running \033[0m")
    # debugging all tensors shapes
    # print(f"\033[96m[DEBUG] {output_t_sp.shape = }\033[0m")
    # print(f"\033[96m[DEBUG] {output_t_attn.shape = }\033[0m")
    # print(f"\033[96m[DEBUG] inp_t.shape: {inp_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] input_offset_t.shape: {input_offset_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] sp_weight_offset_t.shape: {sp_weight_offset_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] sp_weight_scale_t.shape: {sp_weight_scale_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] sp_fp32_bias_t.shape: {sp_fp32_bias_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] attn_weight_offset_t.shape: {attn_weight_offset_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] attn_weight_scale_t.shape: {attn_weight_scale_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] attn_fp32_bias_t.shape: {attn_fp32_bias_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] int8_sp_weight_t.shape: {int8_sp_weight_t.shape}\033[0m")
    # print(f"\033[96m[DEBUG] int8_attn_weight_t.shape: {int8_attn_weight_t.shape}\033[0m")
    
    
    
    
    use_int8_mul = False
    if use_int8_mul:
        # RuntimeError: Pytorch does not support `@` operator for int8 type tensor, throwing "addmm_cuda" not implemented for 'Char'
        tmp_int8_sp = (inp_t - input_offset_t) @ (int8_sp_weight_t - sp_weight_offset_t).transpose(0, 1) # [1, 300, 192]
        tmp_fp32_sp = (tmp_int8_sp) * (sp_weight_scale_t * input_scale_t)
        tmp_fp32_sp = tmp_fp32_sp + sp_fp32_bias_t
        output_t_sp.copy_(tmp_fp32_sp)

        # compute output of attention weights
        tmp_int8_attn = (inp_t - input_offset_t) @ (int8_attn_weight_t - attn_weight_offset_t).transpose(0, 1) # [1, 300, 96]
        tmp_fp32_attn = (tmp_int8_attn) * (attn_weight_scale_t * input_scale_t)
        tmp_fp32_attn = tmp_fp32_attn + attn_fp32_bias_t
        output_t_attn.copy_(tmp_fp32_attn)
    else:
        # convert back to fp32
        sp_fp32_weight = (int8_sp_weight_t - sp_weight_offset_t) * sp_weight_scale_t # (192, 256)
        attn_fp32_weight = (int8_attn_weight_t - attn_weight_offset_t) * attn_weight_scale_t # (96, 256)
        # print(sp_fp32_weight)
        # print(attn_fp32_weight)
        # print(attn_fp32_bias_t)
        input_fp32 = (inp_t - input_offset_t) * input_scale_t
        tmp_fp32_sp = input_fp32 @ sp_fp32_weight.transpose(0, 1) + sp_fp32_bias_t
        tmp_fp32_attn = input_fp32 @ attn_fp32_weight.transpose(0, 1) + attn_fp32_bias_t
        output_t_sp.copy_(tmp_fp32_sp)
        output_t_attn.copy_(tmp_fp32_attn)