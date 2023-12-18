import logging
import math
import os
from typing import Any, Dict, List, Literal, Optional, Type, Union

from solver.base_solver import BaseSolver
from utils.base_param import BaseParam
from utils.my_logger import get_logger

# SolutionT = Optional[Dict[str, Any]]
SolutionT = Optional[Dict[str, float]]
ResultT = Optional[BaseParam]

import numpy as np
from solver.gpkit_solver import GpkitSolver
from solver.scip_solver import ScipSolver
from utils.dse_var import DSEVar
from utils.dse_constr import DSEConstr
from solver.exhaust_solver import ExhaustSolver


class BaseFormulation:
    logger: logging.Logger
    name: str
    path: str
    config: Dict[str, Any]
    # var_dict: Dict[str, DSEVar]
    var_list: List[DSEVar]
    constr_list: List[DSEConstr]
    solution: SolutionT
    result: ResultT

    def __init__(
        self,
        name: str,
        path: str,
        config: Dict[str, Any],
    ) -> None:
        """Base class for DSE formulation.

        build_var_list() and build_constr_list() will be called in __init__().
        Init class variables before calling super().__init__().
        """
        assert os.path.isdir(path)
        self.logger = get_logger(self.__class__.__name__)
        # self.logger =
        self.name = name
        self.path = path
        config.setdefault("solver-flow", "rnr")  # "rnr", "gpkit", "scip", "exh"
        config.setdefault("rnr-method", "exp")  # "lin" or "exp"
        config.setdefault("rnr-range", 2)
        self.config = config
        # self.var_dict = {}
        self.var_list = []
        self.build_var_list()
        self.constr_list = []
        self.build_constr_list()
        self.solution = None
        self.result = None

    # def build_var_dict(self) -> None:
    #     raise NotImplementedError(f"Overriding is requried.")

    def build_var_list(self) -> None:
        raise NotImplementedError(
            f"Overriding of {self.__class__.__name__}.build_var_list() is requried."
        )

    def build_constr_list(self) -> None:
        raise NotImplementedError(
            f"Overriding of {self.__class__.__name__}.build_constr_list() is requried."
        )

    def build_result(self) -> None:
        """Build result from solution.

        Set self.result to `BaseParam`.
        """
        raise NotImplementedError(
            f"Overriding of {self.__class__.__name__}.build_result() is requried."
        )

    def solve(self) -> None:
        if self.config["solver-flow"] == "rnr":
            solution = self.solve_rnr()
        elif self.config["solver-flow"] == "gpkit":
            solution = self.solve_with(GpkitSolver, "gpkit")
        elif self.config["solver-flow"] == "scip":
            solution = self.solve_with(ScipSolver, "scip")
        elif self.config["solver-flow"] == "exh":
            solution = self.solve_with(ExhaustSolver, "exh")
        else:
            raise ValueError(f"flow: {self.config['solver-flow']} is not supported.")
        self.solution = solution
        self.build_result()

    def solve_rnr(self) -> SolutionT:
        # GpkitSolver
        gpkit_solution = self.solve_with(GpkitSolver, "gpkit")
        if gpkit_solution is None:
            return None
        # rounding bound
        self.round_bound(gpkit_solution)
        # ScipSolver
        scip_solution = self.solve_with(ScipSolver, "scip")
        return scip_solution

    def solve_with(self, solver_cls: Type[BaseSolver], postfix: str) -> SolutionT:
        self.logger.info(f"{solver_cls.__name__}: start.")
        solver = solver_cls(
            f"{self.name}-{postfix}",
            self.path,
            self.var_list,
            self.constr_list,
            self.config,
        )
        solver.solve()
        if solver.solution is not None:
            self.logger.debug(solver.solution)
            self.logger.info(
                f"{solver_cls.__name__}: status={solver.status}, obj={solver.solution['obj']}"
            )
        else:
            self.logger.error(f"{solver_cls.__name__} status={solver.status}.")
        return solver.solution

    def round_bound(self, var_val: SolutionT) -> None:
        b_range = self.config["rnr-range"]
        method = self.config["rnr-method"]
        if var_val is None:
            self.logger.warning(f"var_val is None.")
            return
        self.logger.info(f"Rounding bound with {method}(R = {b_range}).")
        for var in self.var_list:
            if var.is_strict:
                for dir, cmp_f in zip(["lb", "ub"], [max, min]):
                    val = var_val[var.name]
                    old_b = getattr(var, dir)
                    new_b = new_bound(val, b_range, dir, method)
                    new_b = new_b if old_b is None else cmp_f(old_b, new_b)
                    if new_b <= 2 and dir == "ub":
                        # for an integer variable, ub < 3-1e-5 will cause
                        # segmentation fault (Signals.SIGSEGV: 11) during presolve
                        # unbelievable!
                        new_b = 3
                    if old_b != new_b:
                        setattr(var, dir, new_b)
                        self.logger.debug(f"{var.name}.{dir}: {old_b} -> {new_b}")


def new_bound(val, b_range, dir, method) -> float:
    if method == "lin":
        return math.floor(val + b_range) if dir == "ub" else math.ceil(val - b_range)
    elif method == "exp":
        return math.floor(val * b_range) if dir == "ub" else math.ceil(val / b_range)
    else:
        raise ValueError(f"method: {method} is not supported.")
