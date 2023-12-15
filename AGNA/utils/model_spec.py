from typing import Any, Dict, Iterator, List
from utils.common import csv_row
from utils.base_param import BaseParam
from utils.node_param import NodeParam

model_spec_fmt = {
    "": "{o.name}: num_nodes={o.num_nodes}, {o.nodes}",
}


class ModelSpec(BaseParam):

    name: str
    nodes: List[NodeParam]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @staticmethod
    def get_param_keys() -> List[str]:
        return [
            "name",
            "nodes",
        ]

    def set_params(self, params: Dict[str, Any]) -> None:
        params.setdefault("nodes", [])
        params["nodes"] = [
            n if isinstance(n, NodeParam) else NodeParam(**n) for n in params["nodes"]
        ]
        return super().set_params(params)

    def unique_node_iter(self) -> Iterator[NodeParam]:
        for node in self.nodes:
            if node.unique_cnt >= 1:
                yield node

    def __format__(self, __format_spec: str) -> str:
        if __format_spec == "csv":
            csv_rows = [f"{node:r}" for node in self.nodes]
            csv_rows.insert(0, f"{self.nodes[0]:h}")
            return "\n".join(csv_rows)
        else:
            return self.name
