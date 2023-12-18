import logging
from typing import Any, Dict, List
from utils.base_param import BaseParam
from utils.common import my_prod
from utils.model_spec import ModelSpec
from utils.my_logger import get_logger
from utils.node_param import NodeParam
from utils.platform_spec import PlatformSpec

agna_dims = list(range(6))
"""0, 1, 2, 3, 4, 5 (K, C, I, J, H, W)
"""

agna_levels = ["T", "S", "P", "Q", "F"]
"""T, S, P, Q, F
"""


class AGNANodeEvaluation:

    logger: logging.Logger
    platform: PlatformSpec
    node: NodeParam

    def __init__(self, platform: PlatformSpec, node: NodeParam) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.platform = platform
        self.node = node

    def get_c_comm(self) -> int:
        c_size = self.node.input_size + self.node.weight_size + self.node.output_size
        c_width = self.platform.dbus_width
        d_width = self.platform.data_width
        self.logger.debug(f"get_c_comm({self.node.name})")
        self.logger.debug(f"  input: {self.node.input_shape} {self.node.input_size}")
        self.logger.debug(f"  weight: {self.node.weight_shape} {self.node.weight_size}")
        self.logger.debug(f"  output: {self.node.output_shape} {self.node.output_size}")
        self.logger.debug(f"  c_size: {c_size}")
        self.logger.debug(f"  return: {(c_size * d_width + c_width - 1) // c_width}")
        return (c_size * d_width + c_width - 1) // c_width


class AGNAConfig(BaseParam):

    dsp_pack: bool
    bram_ovhd: int
    skip_s: bool

    @staticmethod
    def get_param_keys() -> List[str]:
        return [
            "dsp_pack",
            "bram_ovhd",
            "skip_s",
        ]

    def set_params(self, params: Dict[str, Any]) -> None:
        params.setdefault("dsp_pack", True)
        params.setdefault("bram_ovhd", 90)
        params.setdefault("skip_s", True)
        return super().set_params(params)
