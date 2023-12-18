from formulation.base_formulation import BaseFormulation
from utils.dse_var import DSEVar
from utils.dse_constr import DSEConstr
from utils.simp_param import SimpParam


class SimpFormulation(BaseFormulation):
    def build_var_list(self) -> None:
        var = DSEVar("a", 1, 20, is_strict=True)
        self.var_list.append(var)
        var = DSEVar("b", 1, 20, is_strict=True)
        self.var_list.append(var)
        # var = DSEVar("apb", is_inter=True)
        # self.var_list.append(var)
        var = DSEVar("obj", is_obj=True)
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
        constr = DSEConstr("c2", lambda vd: vd["obj"] >= vd["a"] + vd["b"])
        # constr = DSEConstr("c2", lambda vd: vd["c"] == vd["a"])
        # constr = DSEConstr("c2", lambda vd: vd["a"] + vd["b"], 2)
        self.constr_list.append(constr)

    def build_result(self) -> None:
        if self.solution is not None:
            self.result = SimpParam(**self.solution)
