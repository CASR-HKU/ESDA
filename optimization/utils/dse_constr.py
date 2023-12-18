from typing import Any, Callable, Dict, Literal, Optional, Union

VauleT = Union[int, float, None]
ScopeT = Literal["all", "gpkit", "scip", "exhaust"]
VtypeT = Literal["C", "I", "B"]
ExprT = Callable[[Dict[str, Any]], Any]
IntervarT = Union[str, int, float, None]


class DSEConstr:

    name: str
    expression: ExprT
    inter_var: IntervarT
    scope: ScopeT
    # vtype: VtypeT

    @property
    def in_gpkit(self) -> bool:
        return self.scope in ["all", "gpkit"]

    @property
    def in_scip(self) -> bool:
        return self.scope in ["all", "scip"]

    @property
    def in_exhaust(self) -> bool:
        return self.scope in ["all", "exhaust"]

    def __init__(
        self,
        name: str,
        expression: ExprT,
        inter_var: IntervarT = None,
        scope: ScopeT = "all",
    ) -> None:
        self.name = name
        self.expression = expression
        self.inter_var = inter_var
        self.scope = scope

    def get_constr_expr(self, var_dict: Dict[str, Any]) -> Any:
        if isinstance(self.inter_var, str):
            constr_expr = var_dict[self.inter_var] == self.expression(var_dict)
        elif isinstance(self.inter_var, (int, float)):
            constr_expr = self.inter_var == self.expression(var_dict)
        elif self.inter_var is None:
            constr_expr = self.expression(var_dict)
        else:
            raise ValueError(f"Invalid inter_var: {self.inter_var}")
        return constr_expr

    def __format__(self, __format_spec: str) -> str:
        return f"{self.name}, inter_var={self.inter_var}, scope={self.scope}"
