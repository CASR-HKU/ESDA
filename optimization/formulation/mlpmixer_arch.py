from typing import List
from formulation.base_formulation import BaseFormulation
from utils.dse_constr import DSEConstr
from utils.dse_var import DSEVar
from utils.model_spec import ModelSpec
from utils.platform_spec import PlatformSpec


class ArchSearch(BaseFormulation):

    platform: PlatformSpec
    model_list: List[ModelSpec]

    def __init__(
        self, name: str, path: str, platform: PlatformSpec, model_list: List[ModelSpec]
    ) -> None:
        super().__init__(name, path)
        self.platform = platform
        self.model_list = model_list

    def build_var_list(self) -> None:
        var = DSEVar("a", 1, 20, is_strict=True)
        self.var_list.append(var)
        var = DSEVar("b", 1, 20, is_strict=True)
        self.var_list.append(var)
        # var = DSEVar("apb", is_inter=True)
        # self.var_list.append(var)
        var = DSEVar("c", is_obj=True)
        self.var_list.append(var)

    def build_constr_list(self) -> None:
        constr = DSEConstr(
            "c1",
            lambda vd: vd["a"] ** 2
            + vd["a"] * vd["b"]
            + vd["b"] ** 2
            + vd["a"]
            + vd["b"]
            <= 30,
        )
        self.constr_list.append(constr)
        # constr = DSEConstr("c2", lambda vd: vd["a"] + vd["b"], inter_var="c")
        constr = DSEConstr("c2", lambda vd: vd["c"] >= vd["a"] + vd["b"])
        self.constr_list.append(constr)
