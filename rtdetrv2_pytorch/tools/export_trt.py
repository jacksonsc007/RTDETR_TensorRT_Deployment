import os
import argparse
import tensorrt as trt

# import custom plugins
import sys
sys.path.append(".")
# Load if we intend to use python plugins
# import ink_plugins.ink_plugins_decorator 
# import ink_plugins.ink_pluginsIPluginV2 
import ink_plugins.ink_plugins_IPluginV3 


# debug = True
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

def main(onnx_path, engine_path, max_batchsize, opt_batchsize, min_batchsize, plugin_libs, use_fp16, verbose)->None:
    """ Convert ONNX model to TensorRT engine.
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path to save the output TensorRT engine.
        use_fp16 (bool): Whether to use FP16 precision.
        verbose (bool): Whether to enable verbose logging.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # NOTE: load custom plugin lib
    print(f"\033[96m[INFO] Loading custom plugin libraries: {plugin_libs}\033[0m ")
    config.plugins_to_serialize = plugin_libs
    for lib in plugin_libs:
        builder.get_plugin_registry().load_library(lib)
        # NOTE: It does not work if we modify the config.plugins_to_serialize, still empty
        # config.plugins_to_serialize.append(
        #     lib
        # )
    assert len(config.plugins_to_serialize) == len(plugin_libs), \
    f"Failed to serialize plugin libraries. config.plugins_to_serialize: {config.plugins_to_serialize}, plugin_libs: {plugin_libs}"

    # enable verbose profiling
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    print(f"\033[96m[Loading ONNX file from {onnx_path}]\033[0m ")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GiB
    # config.max_workspace_size = 1 << 30  # 1GB
    
    # debug
    config.set_flag(trt.BuilderFlag.DEBUG)
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"\033[96m[FP16 enabled]  \033[0m")
        else:
            print("[WARNING] FP16 not supported on this platform. Proceeding with FP32.")

    profile = builder.create_optimization_profile()
    # profile.set_shape("images", min=(min_batchsize, 3, 640, 640), opt=(opt_batchsize, 3, 640, 640), max=(max_batchsize, 3, 640, 640))
    profile.set_shape("images", min=(1, 3, 640, 640), opt=(1, 3, 640, 640), max=(1, 3, 640, 640))
    profile.set_shape("orig_target_sizes", min=(1, 2), opt=(1, 2), max=(1, 2))
    config.add_optimization_profile(profile)

    print("\033[96m[Building TensorRT engine...]  \033[0m")
    # engine = builder.build_engine(network, config)
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Failed to build the engine.")

    print("\033[96m[Saving engine]  \033[0m")
    with open(engine_path, "wb") as f:
        f.write(engine)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model file")
    parser.add_argument("--saveEngine", "-o", type=str, default="model.engine", help="Path to output TensorRT engine file")
    parser.add_argument("--maxBatchSize", "-Mb", type=int, default=32, help="Maximum batch size for inference")
    parser.add_argument("--optBatchSize", "-ob", type=int, default=16, help="Optimal batch size for inference")
    parser.add_argument("--minBatchSize", "-mb", type=int, default=1, help="Minimum batch size for inference")
    parser.add_argument("--plugin_libs", type=str, nargs='*', default=[], help="List of plugin library paths")
    parser.add_argument("--fp16", default=False, action="store_true", help="Enable FP16 precision mode")
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    main(
        onnx_path=args.onnx,
        engine_path=args.saveEngine,
        max_batchsize=args.maxBatchSize,
        opt_batchsize=args.optBatchSize,
        min_batchsize=args.minBatchSize,
        plugin_libs=args.plugin_libs,
        use_fp16=args.fp16,
        verbose=args.verbose
    )