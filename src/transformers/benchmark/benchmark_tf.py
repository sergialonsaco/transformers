# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
"""
    Benchmarking the library on inference and training in PyTorch.
"""


import logging
import random
import timeit

from transformers import (
    TF_MODEL_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    PretrainedConfig,
    is_py3nvml_available,
    is_tf_available,
)

from .benchmark_utils import (
    Benchmark,
    Memory,
    get_cpu_memory,
    get_gpu_memory,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)


if is_tf_available():
    import tensorflow as tf
    from tensorflow.python.eager import context as tf_context
    from .benchmark_args_tf import TensorflowBenchmarkArguments


if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

logger = logging.getLogger(__name__)


def random_input_ids(batch_size, sequence_length, vocab_size):
    rng = random.Random()

    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]

    return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)


class TensorflowBenchmark(Benchmark):

    args: TensorflowBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "Tensorflow"

    @property
    def framework_version(self):
        return tf.__version__

    def train(self, model_name, batch_size, sequence_length, trace_memory=False):
        try:
            config = self.config_dict[model_name]

            model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

            # encoder-decoder has vocab size saved differently
            vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size

            input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

            def compute_loss_and_backprob_encoder():
                loss = model(input_ids, labels=input_ids, training=True)[0]
                gradients = tf.gradients(loss, model.trainable_variables)
                gradients = None
                return gradients

            def compute_loss_and_backprob_encoder_decoder():
                loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
                gradients = tf.gradients(loss, model.trainable_variables)
                gradients = None
                return gradients

            _train = (
                compute_loss_and_backprob_encoder_decoder
                if config.is_encoder_decoder
                else compute_loss_and_backprob_encoder
            )

            if trace_memory is True:
                if self.args.trace_memory_line_by_line:
                    trace = start_memory_tracing("transformers")

                if self.args.n_gpu > 0:
                    # gpu
                    # clear gpu cache
                    tf_context.context()._clear_caches()  # See https://github.com/tensorflow/tensorflow/issues/20218#issuecomment-416771802

                    # calculate loss and do backpropagation
                    _train()
                elif not self.args.no_tpu and self.args.is_tpu:
                    # tpu
                    raise NotImplementedError(
                        "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.no_memory=True`"
                    )
                else:
                    # cpu
                    memory_bytes = measure_peak_memory_cpu(_train, get_cpu_memory)
                    memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

                if self.args.trace_memory_line_by_line:
                    summary = stop_memory_tracing(trace)
                else:
                    summary = None

                if self.args.n_gpu > 0:
                    # gpu
                    if not is_py3nvml_available():
                        logger.warning(
                            "py3nvml not installed, we won't log GPU memory usage. "
                            "Install py3nvml (pip install py3nvml) to log information about GPU."
                        )
                        memory = "N/A"
                    else:
                        nvml.nvmlInit()
                        max_bytes_in_use = measure_peak_memory_cpu(
                            _train, get_gpu_memory, device_idx=self.arsg.device_idx
                        )
                        nvml.nvmlShutdown()

                        memory = Memory(max_bytes_in_use)

                return memory, summary
            else:
                if not self.args.no_tpu and self.args.is_tpu:
                    # run additional 10 times to stabilize compilation for tpu and torchscript
                    logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                    timeit.repeat(
                        _train, repeat=1, number=5,
                    )

                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(_train, repeat=self.args.repeat, number=10,)

                return min(runtimes) / 10.0
        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            if trace_memory:
                return "N/A", None
            else:
                return "N/A"

    def inference(self, model_name, batch_size, sequence_length, trace_memory=False):
        try:
            config = self.config_dict[model_name]
            model = None

            if self.args.torchscript:
                config.torchscript = True

            if self.args.with_lm_head:
                model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
            else:
                model = TF_MODEL_MAPPING[config.__class__](config)

            # encoder-decoder has vocab size saved differently
            vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size

            input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

            def encoder_decoder_forward():
                model(input_ids, decoder_input_ids=input_ids)

            def encoder_forward():
                model(input_ids)

            _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward

            if trace_memory is True:
                if self.args.trace_memory_line_by_line:
                    trace = start_memory_tracing("transformers")

                if self.args.n_gpu > 0:
                    # gpu
                    # clear gpu cache
                    tf_context.context()._clear_caches()  # See https://github.com/tensorflow/tensorflow/issues/20218#issuecomment-416771802
                    # run forward
                    _forward()
                elif not self.args.no_tpu and self.args.is_tpu:
                    # tpu
                    raise NotImplementedError(
                        "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.no_memory=True`"
                    )
                else:
                    # cpu
                    memory_bytes = measure_peak_memory_cpu(_forward, get_cpu_memory)
                    memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

                if self.args.trace_memory_line_by_line:
                    summary = stop_memory_tracing(trace)
                else:
                    summary = None

                if self.args.n_gpu > 0:
                    # gpu
                    if not is_py3nvml_available():
                        logger.warning(
                            "py3nvml not installed, we won't log GPU memory usage. "
                            "Install py3nvml (pip install py3nvml) to log information about GPU."
                        )
                        memory = "N/A"
                    else:
                        nvml.nvmlInit()
                        max_bytes_in_use = measure_peak_memory_cpu(
                            _forward, get_gpu_memory, device_idx=self.arsg.device_idx
                        )
                        nvml.nvmlShutdown()

                        memory = Memory(max_bytes_in_use)

                return memory, summary
            else:

                if not self.args.no_tpu and not self.args.is_tpu:
                    # run additional 10 times to stabilize compilation for tpu and torchscript
                    logger.info("Do inference on TPU. Running model 5 times to stabilize compilation")
                    timeit.repeat(
                        _forward, repeat=1, number=5,
                    )

                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(_forward, repeat=self.args.repeat, number=10,)

                return min(runtimes) / 10.0

        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            if trace_memory:
                return "N/A", None
            else:
                return "N/A"
