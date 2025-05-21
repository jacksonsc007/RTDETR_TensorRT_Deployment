import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import numpy as np
import re
import argparse
import copy
import sys
sys.path.append(".")
import ctypes

debug = False
if debug:
    # improve torch tensor printing
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    # debug with debugpy
    import debugpy
    # Listen on a specific port (choose any available port, e.g., 61074)
    debugpy.listen(("0.0.0.0", 61074))
    print("Waiting for debugger to attach...")
    # Optional: Wait for the debugger to attach before continuing execution
    debugpy.wait_for_client()

# Parse command line arguments
parser = argparse.ArgumentParser(description='ONNX Graph Surgeon for RTDETR')
parser.add_argument('--onnx_file', type=str, required=True, help='Path to the input ONNX file')
parser.add_argument('--plugin_path', type=str, required=False, help='the cpp plugin library', default="")
parser.add_argument('--output_file', type=str, help='Path to save the modified ONNX file (default: graph_surgeon-<input_file>)')
parser.add_argument('--verbose', action="store_true", default=False, help="enable verbose logging for TensorRT engine building")
parser.add_argument('--build_engine', action="store_true", default=False, )
args = parser.parse_args()

onnxFile = args.onnx_file
orignal_graph = gs.import_onnx(onnx.load(onnxFile))
graph = orignal_graph.copy()

namespace = "ink_plugins"
plugin_name = "fused_attn_offset_prediction" # operation name
node_name_template = "FusedAttnOffsetPrediction_layer{}" # node name template

# Find all decoder layers
layer_pattern = re.compile(r"/model/decoder/decoder/layers\.(\d+)/")
layer_indices = set()

for node in graph.nodes:
    match = layer_pattern.search(node.name)
    if match:
        layer_indices.add(int(match.group(1)))

layer_indices = sorted(list(layer_indices))
print(f"Found {len(layer_indices)} decoder layers: {layer_indices}")

# Process each layer
# for layer_idx in range(1):
for layer_idx in layer_indices:
    layer_prefix = f"/model/decoder/decoder/layers.{layer_idx}"
    
    start_node = None
    attn_end_node = None
    sp_offset_end_node = None
    sp_weight_dq_node = None
    sp_weight_q_node = None
    sp_bias_node = None
    attn_weight_dq_node = None
    attn_weight_q_node = None
    attn_bias_node = None
    
    for node in graph.nodes:
        if node.name == f"{layer_prefix}/cross_attn/sampling_offsets/input_quantizer/DequantizeLinear":
            # print(f"Found start node for layer {layer_idx}: {node.name}")
            start_node = node
        if node.name == f"{layer_prefix}/cross_attn/sampling_offsets/Add":
            sp_offset_end_node = node
        if node.name == f"{layer_prefix}/cross_attn/attention_weights/Add":
            attn_end_node = node
        if node.name == f"{layer_prefix}/cross_attn/sampling_offsets/weight_quantizer/DequantizeLinear":
            sp_weight_dq_node = node
        if node.name == f"{layer_prefix}/cross_attn/attention_weights/weight_quantizer/DequantizeLinear":
            attn_weight_dq_node = node
        if node.name == f"{layer_prefix}/cross_attn/sampling_offsets/weight_quantizer/QuantizeLinear":
            sp_weight_q_node = node
        if node.name == f"{layer_prefix}/cross_attn/attention_weights/weight_quantizer/QuantizeLinear":
            attn_weight_q_node = node
        if node.name == f"{layer_prefix}/cross_attn/attention_weights/Add":
            attn_bias_node = node
        if node.name == f"{layer_prefix}/cross_attn/sampling_offsets/Add":
            sp_bias_node = node
    
    # Skip if we couldn't find all required nodes for this layer
    if not all([start_node, attn_end_node, sp_offset_end_node, sp_weight_dq_node, 
                sp_weight_q_node, sp_bias_node, attn_weight_dq_node, 
                attn_weight_q_node, attn_bias_node]):
        raise ValueError(f"Skipping layer {layer_idx} - couldn't find all required nodes")

    # bias of linear layer
    sp_fp32_bias = sp_bias_node.inputs[0].values
    attn_fp32_bias = attn_bias_node.inputs[0].values

    # manually quantize fp32 weight to int8
    def quantize_fp32_to_int8(q_node):
        w_fp32 = q_node.inputs[0].values
        scale = q_node.inputs[1].values
        scale = scale[:, np.newaxis].astype(np.float32)
        zero_point = q_node.inputs[2].values
        zero_point = zero_point[:, np.newaxis].astype(np.float32)
        # assert zero_point is 0
        assert np.allclose(zero_point, 0)
        w_int8 = np.round(w_fp32 / scale) + zero_point
        w_int8 = np.clip(w_int8, -128, 127).astype(np.int8)
        return w_int8
        
    int8_sp_weight = quantize_fp32_to_int8(sp_weight_q_node)
    int8_attn_weight = quantize_fp32_to_int8(attn_weight_q_node)

    pluginNode_input = start_node.inputs[0]

    # save scaling factors and offsets as attributes of pluginNode
    input_scale = start_node.inputs[1].values
    input_offset = start_node.inputs[2].values

    # quantization metadata
    sp_weight_scale = sp_weight_dq_node.inputs[1].values
    sp_weight_offset = sp_weight_dq_node.inputs[2].values
    attn_weight_scale = attn_weight_dq_node.inputs[1].values
    attn_weight_offset = attn_weight_dq_node.inputs[2].values

    # Create unique output variable names for this layer
    plugin_output_attn = gs.Variable(f"FusedPredictionsAttnSamplingOffsets_attn_layer{layer_idx}_decoder", 
                                     np.dtype(np.float32), (1, 300, 96)) 
    plugin_output_sp = gs.Variable(f"FusedPredictionsAttnSamplingOffsets_sp_layer{layer_idx}_decoder", 
                                   np.dtype(np.float32), (1, 300, 192)) 

    # avoid 0-d shape
    input_scale = input_scale.reshape(1)
    input_offset = input_offset.reshape(1)

    """
    gs.Constant is used to wrap multi-dimensional constants.
    """
    attrs = {
            "input_scale": gs.Constant(name=f"input_scale", values=input_scale),
            "input_offset": gs.Constant(name=f"input_offset", values=input_offset),
            "sp_weight_scale": gs.Constant(name=f"sp_weight_scale", values=sp_weight_scale),
            "sp_weight_offset": gs.Constant(name=f"sp_weight_offset", values=sp_weight_offset),
            "attn_weight_scale": gs.Constant(name=f"attn_weight_scale", values=attn_weight_scale),
            "attn_weight_offset": gs.Constant(name=f"attn_weight_offset", values=attn_weight_offset),
            "int8_sp_weight": gs.Constant(name=f"int8_sp_weight", values=int8_sp_weight),
            "int8_attn_weight": gs.Constant(name=f"int8_attn_weight", values=int8_attn_weight),
            "sp_fp32_bias": gs.Constant(name=f"sp_fp32_bias", values=sp_fp32_bias),
            "attn_fp32_bias": gs.Constant(name=f"attn_fp32_bias", values=attn_fp32_bias),
            "plugin_namespace": namespace
    }
    """
    Save all attributes seperately as binary files
    """
    if layer_idx == 0:
        for attr_name, attr_value in attrs.items():
            if (attr_name != "plugin_namespace"):
                attr_value_path = f"{attr_name}.bin"
                # attr_value_path = f"{attr_name}_layer{layer_idx}.bin"
                with open(attr_value_path, "wb") as f:
                    f.write(attr_value.values.tobytes())
                print(f"Saved {attr_name} to {attr_value_path}")
        

    node_name = node_name_template.format(layer_idx)
    pluginNode = gs.Node(
        plugin_name,
        node_name,
        inputs = [pluginNode_input],
        outputs= [plugin_output_attn, plugin_output_sp],
        attrs=attrs)

    graph.nodes.append(pluginNode)
    print(f"Added plugin node for layer {layer_idx}: {node_name}")

    # Reconnect the graph
    sp_end_next_node = sp_offset_end_node.o(0)
    sp_end_next_node.inputs[0] = plugin_output_sp

    attn_end_next_node = attn_end_node.o(0)
    attn_end_next_node.inputs[0] = plugin_output_attn

    # Disconnect original nodes
    start_node.inputs.clear()
    sp_offset_end_node.outputs.clear()
    attn_end_node.outputs.clear()
    nodes_to_remove = [start_node, sp_weight_dq_node, sp_weight_q_node, sp_bias_node,
                       attn_weight_dq_node, attn_weight_q_node, attn_bias_node,
                       sp_offset_end_node, attn_end_node]

    for node in nodes_to_remove:
        if node in graph.nodes:
            graph.nodes.remove(node)
# Clean up the graph and save
graph.cleanup().toposort()
model = gs.export_onnx(graph)

# Determine output file path
if args.output_file:
    surgeoned_onnx_path = args.output_file
else:
    surgeoned_onnx_path = "graph_surgeon-" + onnxFile.split('/')[-1]

# check onnx model
# onnx.checker.check_model(model)
onnx.save(model, surgeoned_onnx_path)
print(f"Saved modified model to {surgeoned_onnx_path}")
