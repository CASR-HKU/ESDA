from copy import deepcopy
from typing import Any, Dict, List
from utils.dse_var import DSEVar
from utils.dse_constr import DSEConstr
from solver.base_solver import BaseSolver


class ExhaustSolver(BaseSolver):

    var_list: List[DSEVar]
    constr_list: List[DSEConstr]

    def build(self, var_list: List[DSEVar], constr_list: List[DSEConstr]) -> None:
        self.var_list = deepcopy(var_list)
        self.constr_list = deepcopy(constr_list)
        
        