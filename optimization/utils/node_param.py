from typing import Any, Dict, List, Tuple
from utils.common import csv_row, my_prod
from utils.base_param import BaseParam

node_param_fmt = {
    "": "{o.name}: type={o.type}, L={o.loop_bounds}",
    "csv": "{o.name}, {o.unique_cnt}, {o.type}, {o.loop_bounds}",
}


class NodeParam(BaseParam):

    name: str
    unique_cnt: int
    same_as: str
    input_source: str
    add_source: str
    type: int
    loop_bounds: List[int]
    strides: List[int]
    paddings: List[int]
    use_bn: bool
    use_add: bool
    use_act: bool

    @property
    def is_dpws(self) -> bool:
        return self.type >= 1

    @property
    def use_weight(self) -> bool:
        return self.type <= 1

    @property
    def input_shape(self) -> List[int]:
        c = self.loop_bounds[1]
        ih = (
            self.loop_bounds[4] * self.strides[0]
            + self.loop_bounds[2]
            - 1
            - self.paddings[0]
            - self.paddings[1]
        )
        iw = (
            self.loop_bounds[5] * self.strides[1]
            + self.loop_bounds[3]
            - 1
            - self.paddings[2]
            - self.paddings[3]
        )
        return [c, ih, iw]

    @property
    def weight_shape(self) -> List[int]:
        return [self.loop_bounds[i] for i in range(4)]

    @property
    def output_shape(self) -> List[int]:
        k = self.loop_bounds[1] if self.is_dpws else self.loop_bounds[0]
        h = self.loop_bounds[4]
        w = self.loop_bounds[5]
        return [k, h, w]

    @property
    def input_size(self) -> int:
        return my_prod(self.input_shape)

    @property
    def weight_size(self) -> int:
        return my_prod(self.weight_shape) if self.use_weight else 0

    @property
    def output_size(self) -> int:
        return my_prod(self.output_shape)

    @staticmethod
    def get_param_keys() -> List[str]:
        return [
            "name",
            "unique_cnt",
            "same_as",
            "input_source",
            "add_source",
            "type",
            "loop_bounds",
            "strides",
            "paddings",
            "use_bn",
            "use_add",
            "use_act",
        ]

    def set_params(self, params: Dict[str, Any]) -> None:
        params.setdefault("unique_cnt", 1)
        params.setdefault("same_as")
        params.setdefault("input_source")
        params.setdefault("add_source")
        return super().set_params(params)

    def __format__(self, __format_spec: str) -> str:
        if __format_spec == "h":
            return csv_row(self.get_param_keys())
        elif __format_spec == "r":
            return csv_row(self.get_params())
        else:
            return super().__format__(__format_spec)
