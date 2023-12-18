import logging
from typing import Any, Dict, List, Literal, Optional
from utils.dse_var import DSEVar
from utils.dse_constr import DSEConstr
from utils.my_logger import get_logger

status_T = Literal["init", "fail", "done"]
SolutionT = Optional[Dict[str, float]]


class BaseSolver:
    logger: logging.Logger
    name: str
    path: str
    config: Dict[str, Any]
    solution: SolutionT
    status: status_T

    def __init__(
        self,
        name: str,
        path: str,
        var_list: List[DSEVar],
        constr_list: List[DSEConstr],
        config: Dict[str, Any],
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.name = name
        self.path = path
        self.config = config
        self.solution = None
        self.status = "init"
        self.build(var_list, constr_list)

    def build(self, var_list: List[DSEVar], constr_list: List[DSEConstr]) -> None:
        raise NotImplementedError(f"Overriding is requried.")

    def solve(self) -> None:
        raise NotImplementedError(f"Overriding is requried.")
