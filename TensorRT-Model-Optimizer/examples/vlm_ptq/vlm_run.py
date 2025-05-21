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

import argparse
import os

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner
from utils import add_common_args


def print_result(model, input_text, output_text, args):
    logger.info("---------------------------------------------------------")
    if model.model_type != "nougat":
        logger.info(f"\n[Q] {input_text}")
    for i in range(len(output_text)):
        logger.info(f"\n[A]: {output_text[i]}")

    if args.num_beams == 1:
        output_ids = model.tokenizer(output_text[0][0], add_special_tokens=False)["input_ids"]
        logger.info(f"Generated {len(output_ids)} tokens")

    if args.check_accuracy:
        if model.model_type != "nougat":
            if model.model_type == "vila":
                for i in range(len(args.image_path.split(args.path_sep))):
                    if i % 2 == 0:
                        assert (
                            output_text[i][0].lower()
                            == "the image captures a bustling city intersection teeming with life. "
                            + "from the perspective of a car's dashboard camera, we see"
                        )
                    else:
                        assert (
                            output_text[i][0].lower()
                            == "the image captures the iconic merlion statue in singapore, "
                            + "a renowned worldwide landmark. the merlion, a mythical"
                        )
            elif model.model_type == "llava":
                for i in range(len(args.image_path.split(args.path_sep))):
                    assert output_text[i][0].lower() == "singapore"
            elif model.model_type == "fuyu":
                assert output_text[0][0].lower() == "4"
            elif model.model_type == "pix2struct":
                assert (
                    "characteristic | cat food, day | cat food, wet | cat treats"
                    in output_text[0][0].lower()
                )
            elif model.model_type in ["blip2", "neva", "phi-3-vision", "llava_next"]:
                assert "singapore" in output_text[0][0].lower()
            elif model.model_type == "video-neva":
                assert "robot" in output_text[0][0].lower()
            elif model.model_type == "kosmos-2":
                assert "snowman" in output_text[0][0].lower()
            elif model.model_type == "mllama":
                if "If I had to write a haiku for this one" in input_text:
                    assert (
                        "it would be:.\\nPeter Rabbit is a rabbit.\\nHe lives in a"
                        in output_text[0][0]
                        or "Here is a haiku for the image:\n\n" in output_text[0][0]
                    ), (
                        f"expected results: 'it would be:.\\nPeter Rabbit is a rabbit.\\nHe lives in a', \
                            generated results: '{output_text[0][0]}'"
                    )
                elif "The key to life is" in input_text:
                    assert (
                        "to find your passion and pursue it with all your heart."
                        in output_text[0][0]
                        or "not to be found in the external world," in output_text[0][0]
                    ), (
                        f"expected results: 'to find your passion and pursue it with all your heart.', \
                            generated results: '{output_text[0][0]}'"
                    )
            elif model.model_type == "llava_onevision":
                if args.video_path is None:
                    assert "singapore" in output_text[0][0].lower()
                else:
                    assert (
                        "the video is funny because the child's actions are"
                        in output_text[0][0].lower()
                    )
            elif model.model_type == "qwen2_vl":
                assert "dog" in output_text[0][0].lower()
            else:
                assert output_text[0][0].lower() == "singapore"

    if args.run_profiling:

        def msec_per_batch(name):
            return 1000 * profiler.elapsed_time_in_sec(name) / args.profiling_iterations

        logger.info("Latencies per batch (msec)")
        logger.info("TRT vision encoder: %.1f" % (msec_per_batch("Vision")))
        logger.info("TRTLLM LLM generate: %.1f" % (msec_per_batch("LLM")))
        logger.info("Multimodal generate: %.1f" % (msec_per_batch("Generate")))

    logger.info("---------------------------------------------------------")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()
    logger.set_level(args.log_level)

    model = MultimodalModelRunner(args)
    input_multimodal_data = model.load_test_data(args.image_path, args.video_path)

    num_iters = args.profiling_iterations if args.run_profiling else 1

    for _ in range(num_iters):
        input_text, output_text = model.run(
            args.input_text, input_multimodal_data, args.max_new_tokens
        )

    runtime_rank = tensorrt_llm.mpi_rank()
    if runtime_rank == 0:
        print_result(model, input_text, output_text, args)
