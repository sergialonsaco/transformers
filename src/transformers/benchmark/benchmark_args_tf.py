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

import logging
from dataclasses import dataclass, field
from typing import Tuple

from ..file_utils import cached_property, is_tf_available, tf_required
from .benchmark_args_utils import BenchmarkArguments


if is_tf_available():
    import tensorflow as tf


logger = logging.getLogger(__name__)


@dataclass
class TensorflowBenchmarkArguments(BenchmarkArguments):
    tpu_name: str = field(
        default=None, metadata={"help": "Name of TPU"},
    )
    device_idx: int = field(
        default=0, metadata={"help": "CPU / GPU device index. Defaults to 0."},
    )

    @cached_property
    @tf_required
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        logger.info("Tensorflow: setting up strategy")

        if self.is_tpu:
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)

            strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
        else:
            # currently no multi gpu is allowed
            strategy = tf.distribute.OneDeviceStrategy(device=self.device)

        return strategy

    @property
    @tf_required
    def tpu(self):
        try:
            if self.tpu_name:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self.tpu_name)
            else:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            return tpu
        except ValueError:
            return None

    @property
    @tf_required
    def is_tpu(self):
        return self.tpu is not None and not self.args.no_tpu

    @property
    @tf_required
    def strategy(self) -> "tf.distribute.Strategy":
        return self._setup_strategy

    @property
    @tf_required
    def n_gpu(self) -> int:
        return self._setup_strategy.num_replicas_in_sync

    @property
    def is_gpu(self):
        return self.n_gpu > 0

    @property
    @tf_required
    def device(self) -> str:
        gpus = tf.config.list_physical_devices("GPU")
        if self.is_tpu:
            return self.tpu
        if not self.no_cuda and len(gpus) > 0:
            return "/gpu:{self.device_idx}"  # currently only single device is supported
        else:
            return "/cpu:{self.device_idx}"
