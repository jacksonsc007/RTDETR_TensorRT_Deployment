import tensorrt as trt
import numpy as np
import torch
from polygraphy.json import to_json, from_json
import cupy

class fused_attn_offset_prediction(trt.IPluginV2DynamicExt):
    def __init__(self, fieldCollection=None):
        super(fused_attn_offset_prediction, self).__init__()
        self.num_outputs = 2
        self.plugin_namespace = "ink_plugins"
        self.plugin_version = "1"
        self.plugin_type = "fused_attn_offset_prediction"
        
        # Deserialize weights if provided
        if fieldCollection:
            expected_field_name = ['input_scale', 'input_offset', 'sp_weight_scale', 'sp_weight_offset', 'attn_weight_scale', 'attn_weight_offset', 'int8_sp_weight', 'int8_attn_weight', 'sp_fp32_bias', 'attn_fp32_bias']
            received_field_names = [field.name for field in fieldCollection]
            assert expected_field_name == received_field_names
            self.input_scale = fieldCollection[0].data
            self.input_offset = fieldCollection[1].data
            self.sp_weight_scale = fieldCollection[2].data
            self.sp_weight_offset = fieldCollection[3].data
            self.attn_weight_scale = fieldCollection[4].data
            self.attn_weight_offset = fieldCollection[5].data
            self.int8_sp_weight = fieldCollection[6].data
            self.int8_attn_weight = fieldCollection[7].data
            self.sp_fp32_bias = fieldCollection[8].data
            self.attn_fp32_bias = fieldCollection[9].data

    def configure_plugin(self, inp, out):
        pass

    def get_output_datatype(self, index, input_types):
        if index == 0:
            return trt.float32
        elif index == 1:
            return trt.float32
        else:
            raise ValueError(f"output index {index} is invalid")

    def get_output_dimensions(self, output_idx, inputs, expr_builder):
        # Input shape: [bs, n_queries, 256]
        input_dims =  trt.DimsExprs(inputs[0])
        bs = input_dims[0]
        n_queries = input_dims[1]

        if output_idx == 0:
            return trt.DimsExprs([bs, n_queries, expr_builder.constant(96)])
        else:
            return trt.DimsExprs([bs, n_queries, expr_builder.constant(192)])

    def serialize(self):
        # Serialize all weights and parameters
        field_collections = {
            "input_scale": self.input_scale,
            "input_offset": self.input_offset,
            "sp_weight_scale": self.sp_weight_scale,
            "sp_weight_offset": self.sp_weight_offset,
            "attn_weight_scale": self.attn_weight_scale,
            "attn_weight_offset": self.attn_weight_offset,
            "int8_sp_weight": self.int8_sp_weight,
            "int8_attn_weight": self.int8_attn_weight,
            "sp_fp32_bias": self.sp_fp32_bias,
            "attn_fp32_bias": self.attn_fp32_bias
        }
        return to_json(field_collections)

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert pos < len(in_out)
        d = in_out[pos]
        if d.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return d.type == trt.int8
        else:
            return d.type == trt.float32

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        num_outputs = len(outputs)
        assert num_outputs == self.num_outputs, "Number of outputs does not match expected number of outputs."
        input_dtype = trt.nptype(input_desc[0].type)
        output_attn_dtype = trt.nptype(output_desc[0].type)
        output_offset_dtype = trt.nptype(output_desc[1].type)
        def volume(d):
            return np.prod(d)
        input_mem = cupy.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cupy.dtype(input_dtype).itemsize, self
        )
        output_attn_mem = cupy.cuda.UnownedMemory(
            outputs[0], volume(output_desc[0].dims) * cupy.dtype(output_attn_dtype).itemsize, self
        )
        output_offset_mem = cupy.cuda.UnownedMemory(
            outputs[1], volume(output_desc[1].dims) * cupy.dtype(output_offset_dtype).itemsize, self
        )
        input_ptr = cupy.cuda.MemoryPointer(input_mem, 0)
        output_attn_ptr = cupy.cuda.MemoryPointer(output_attn_mem, 0)
        output_offset_ptr = cupy.cuda.MemoryPointer(output_offset_mem, 0)
        
        input_d = cupy.ndarray(
            tuple(input_desc[0].dims),
            dtype=input_dtype,
            memptr=input_ptr,
        )
        output_attn_d = cupy.ndarray(
            tuple(output_desc[0].dims),
            dtype=output_attn_dtype,
            memptr=output_attn_ptr,
        )
        output_offset_d = cupy.ndarray(
            tuple(output_desc[1].dims),
            dtype=output_offset_dtype,
            memptr=output_offset_ptr,
        )
        
        inp_t = torch.as_tensor(input_d)
        
        # get attributes as tensors
        # Need to reshape since attributes are flattened
        int8_sp_weight_t = torch.as_tensor(self.int8_sp_weight, device="cuda").reshape(192, 256) # [192, 256]
        int8_attn_weight_t = torch.as_tensor(self.int8_attn_weight, device="cuda").reshape(96, 256) # [96, 256]
        sp_fp32_bias_t = torch.as_tensor(self.sp_fp32_bias, device="cuda") # [192]
        attn_fp32_bias_t = torch.as_tensor(self.attn_fp32_bias, device="cuda") # [96]
        input_scale_t = torch.as_tensor(self.input_scale, device="cuda") # [1]
        input_offset_t = torch.as_tensor(self.input_offset, device="cuda") # [1]
        sp_weight_scale_t = torch.as_tensor(self.sp_weight_scale, device="cuda").reshape(192, 1) # [192]
        sp_weight_offset_t = torch.as_tensor(self.sp_weight_offset, device="cuda").reshape(192, 1) # [192]
        attn_weight_scale_t = torch.as_tensor(self.attn_weight_scale, device="cuda").reshape(96, 1) # [96]
        attn_weight_offset_t = torch.as_tensor(self.attn_weight_offset, device="cuda").reshape(96, 1) # [96]
        
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

            # compute output of attention weights
            tmp_int8_attn = (inp_t - input_offset_t) @ (int8_attn_weight_t - attn_weight_offset_t).transpose(0, 1) # [1, 300, 96]
            tmp_fp32_attn = (tmp_int8_attn) * (attn_weight_scale_t * input_scale_t)
            tmp_fp32_attn = tmp_fp32_attn + attn_fp32_bias_t
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
        print("\033[96m[input_int8]\n\033[0m", inp_t)
        # print("\033[96m[attn]\n\033[0m", tmp_fp32_attn)
        # print("\033[96m[sp]\n\033[0m]]", tmp_fp32_sp)
        # save to local file
        # torch.save(inp_t, "plugin_input_python.pt")
        # torch.save(tmp_fp32_attn, "tmp_fp32_attn.pt")
        # torch.save(tmp_fp32_sp, "tmp_fp32_sp.pt")
        # save as binary file to be read by C++
        input_int8_path = "input_int8.bin"
        with open(input_int8_path, "wb") as f:
            f.write(inp_t.cpu().numpy().tobytes())
            print(f"\033[96m[INFO] input_int8_path saved as {input_int8_path}\033[0m")
        output_attn_path = "output_attn.bin"
        with open(output_attn_path, "wb") as f:
            f.write(tmp_fp32_attn.cpu().numpy().tobytes())
            print(f"\033[96m[INFO] output_attn_path saved as {output_attn_path}\033[0m")
        output_sp_path = "output_sp.bin"
        with open(output_sp_path, "wb") as f:
            f.write(tmp_fp32_sp.cpu().numpy().tobytes())
            print(f"\033[96m[INFO] output_sp_path saved as {output_sp_path}\033[0m")
                    
        
        cupy.copyto(
            output_attn_d, 
                cupy.asarray(tmp_fp32_attn),
        )
        cupy.copyto(
            output_offset_d, 
                cupy.asarray(tmp_fp32_sp),
        )

        
        return 0

    # def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
    #     # Convert device pointers to tensors
    #     def from_dev(ptr, shape, dtype):
    #         return torch.from_dlpack(torch.utils.dlpack.toDevicePointer(ptr, tuple(shape), dtype))

    #     inp_shape = (1, 300, 256)
    #     sp_out_shape = (1, 300, 192)
    #     attn_out_shape = (1, 300, 96)

    #     input_tensor = from_dev(inputs[0], inp_shape, torch.int8)

    #     # Convert all weights and biases to tensors on GPU
    #     sp_weight = torch.tensor(self.int8_sp_weight, device="cuda", dtype=torch.int8).view(192, 256)
    #     attn_weight = torch.tensor(self.int8_attn_weight, device="cuda", dtype=torch.int8).view(96, 256)
    #     sp_bias = torch.tensor(self.sp_fp32_bias, device="cuda", dtype=torch.float32)
    #     attn_bias = torch.tensor(self.attn_fp32_bias, device="cuda", dtype=torch.float32)

    #     input_scale = torch.tensor(self.input_scale.item(), device="cuda", dtype=torch.float32)
    #     input_offset = torch.tensor(self.input_offset.item(), device="cuda", dtype=torch.int8)

    #     sp_wscale = torch.tensor(self.sp_weight_scale, device="cuda", dtype=torch.float32).view(192, 1)
    #     sp_woffset = torch.tensor(self.sp_weight_offset, device="cuda", dtype=torch.int8).view(192, 1)

    #     attn_wscale = torch.tensor(self.attn_weight_scale, device="cuda", dtype=torch.float32).view(96, 1)
    #     attn_woffset = torch.tensor(self.attn_weight_offset, device="cuda", dtype=torch.float32).view(96, 1)

    #     # Dequantize input
    #     input_fp32 = (input_tensor.to(torch.float32) - input_offset.to(torch.float32)) * input_scale

    #     # Dequantize weights
    #     sp_weight_fp32 = (sp_weight.to(torch.float32) - sp_woffset.to(torch.float32)) * sp_wscale
    #     attn_weight_fp32 = (attn_weight.to(torch.float32) - attn_woffset.to(torch.float32)) * attn_wscale

    #     # Compute outputs
    #     sp_out = torch.matmul(input_fp32, sp_weight_fp32.t()) + sp_bias
    #     attn_out = torch.matmul(input_fp32, attn_weight_fp32.t()) + attn_bias

    #     # Copy to outputs
    #     sp_out_dl = torch.utils.dlpack.toDlpack(sp_out.contiguous())
    #     attn_out_dl = torch.utils.dlpack.toDlpack(attn_out.contiguous())

    #     out_sp = from_dev(outputs[0], sp_out_shape, torch.float32)
    #     out_attn = from_dev(outputs[1], attn_out_shape, torch.float32)

    #     out_sp.copy_(torch.from_dlpack(sp_out_dl))
    #     out_attn.copy_(torch.from_dlpack(attn_out_dl))

    #     return 0

    def clone(self):
        cloned_plugin = fused_attn_offset_prediction()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def get_output_data_types(self, index, data_type):
        return trt.float32


class fused_attn_offset_prediction_Plugin_Creator(trt.IPluginCreator):
    def __init__(self):
        super(fused_attn_offset_prediction_Plugin_Creator, self).__init__()
        self.name = "fused_attn_offset_prediction"
        self.plugin_namespace = "ink_plugins"
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                trt.PluginField("input_scale", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("input_offset", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("sp_weight_scale", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("sp_weight_offset", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("attn_weight_scale", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("attn_weight_offset", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("int8_sp_weight", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("int8_attn_weight", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("sp_fp32_bias", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("attn_fp32_bias", np.array([]), trt.PluginFieldType.FLOAT32),
            ]
        )

    def create_plugin(self, name, serial_data):
        return fused_attn_offset_prediction(serial_data)

    def deserialize_plugin(self, name, serial_data):
        j = dict(from_json(serial_data.decode("utf-8")))
        deserialized = fused_attn_offset_prediction()
        deserialized.__dict__.update(j)
        return deserialized

# NOTE: manually add plugin
plg_registry = trt.get_plugin_registry()
namespace = "ink_plugins"
# namespace = ""
plg_registry.register_creator(fused_attn_offset_prediction_Plugin_Creator(), namespace)