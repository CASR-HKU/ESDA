import os
from typing import Any, Dict, List
import gpkit
from utils.dse_constr import DSEConstr
from utils.dse_var import DSEVar
from solver.base_solver import BaseSolver


class GpkitSolver(BaseSolver):
    var_name_list: List[str]
    gpkit_model: gpkit.Model

    def __init__(
        self,
        name: str,
        path: str,
        var_list: List[DSEVar],
        constr_list: List[DSEConstr],
        config: Dict[str, Any],
    ) -> None:
        config.setdefault("gpkit-maxnthreads", 8)
        super().__init__(name, path, var_list, constr_list, config)

    def build(self, var_list: List[DSEVar], constr_list: List[DSEConstr]) -> None:
        self.var_name_list = []
        gpkit_var_dict = {}
        gpkit_constr_list = []
        gpkit_obj = None
        # add variables and bounds
        for var in var_list:
            if var.in_gpkit:
                self.logger.debug(f"Adding var: {var}")
                self.var_name_list.append(var.name)
                gpkit_var_dict[var.name] = gpkit.Variable(name=var.name)
                if var.lb is not None:
                    bound_expr = gpkit_var_dict[var.name] >= var.lb
                    assert isinstance(bound_expr, gpkit.nomials.PosynomialInequality)
                    gpkit_constr_list.append(bound_expr)
                if var.ub is not None:
                    bound_expr = gpkit_var_dict[var.name] <= var.ub
                    assert isinstance(bound_expr, gpkit.nomials.PosynomialInequality)
                    gpkit_constr_list.append(bound_expr)
                if var.is_obj:
                    gpkit_obj = gpkit_var_dict[var.name]
        # add constraints
        for constr in constr_list:
            if constr.in_gpkit:
                self.logger.debug(f"Adding constr: {constr}")
                constr_expr = constr.get_constr_expr(gpkit_var_dict)
                # Monomial <= Posynomial, will raise TypeError by gpkit
                # Monomial >= Posynomial
                if isinstance(constr_expr, gpkit.nomials.PosynomialInequality):
                    pass
                # Monomial == Monomial
                elif isinstance(constr_expr, gpkit.nomials.MonomialEquality):
                    pass
                # Monomial == Posynomial, should not be used in gpkit
                elif isinstance(constr_expr, bool):
                    raise TypeError(
                        f"Unsupported constraint: {constr.name} (Monomial == Posynomial)"
                    )
                else:
                    raise TypeError(
                        f"Unknown type of constraint: {constr.name} ({type(constr_expr)})"
                    )
                gpkit_constr_list.append(constr_expr)
        # build model
        assert gpkit_obj is not None, "Objective is not defined."
        self.gpkit_model = gpkit.Model(gpkit_obj, gpkit_constr_list)
        # set path
        self.gpkit_model_path = os.path.join(self.path, f"{self.name}.model")
        self.gpkit_log_path = os.path.join(self.path, f"{self.name}.log")
        self.gpkit_sol_path = os.path.join(self.path, f"{self.name}.sol")
        # save model
        open(self.gpkit_model_path, "w").write(str(self.gpkit_model))

    def solve(self) -> None:
        try:
            # set OMP_NUM_THREADS to gpkit-maxnthreads
            os.environ["OMP_NUM_THREADS"] = str(self.config["gpkit-maxnthreads"])
            gpkit_sol = self.gpkit_model.solve(verbosity=0)
            open(self.gpkit_sol_path, "w").write(gpkit_sol.table())
        except Exception as e:
            self.logger.exception(e)
            self.solution = None
            self.status = "fail"
        else:
            self.solution = dict(
                (k, gpkit_sol["variables"][k]) for k in self.var_name_list
            )
            self.status = "done"
