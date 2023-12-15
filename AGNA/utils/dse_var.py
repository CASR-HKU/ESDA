from typing import Literal, Union

VauleT = Union[int, float, None]
ScopeT = Literal["all", "gpkit", "scip", "exhaust"]
VtypeT = Literal["BINARY", "INTEGER", "CONTINUOUS"]


class DSEVar:
    name: str
    lb: VauleT
    ub: VauleT
    is_obj: bool
    is_strict: bool
    is_inter: bool
    """Intermediate variable."""
    scope: ScopeT
    vtype: VtypeT
    """Variable type."""
    value: VauleT

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
        lb: VauleT = None,
        ub: VauleT = None,
        is_obj=False,
        is_strict=False,
        is_inter=False,
        scope: ScopeT = "all",
        vtype: VtypeT = "INTEGER",
    ) -> None:
        assert (
            not is_obj or name == "obj"
        ), f"objective must be named 'obj' instead of '{name}'"
        self.name = name
        self.lb = lb
        self.ub = ub
        self.is_obj = is_obj
        self.is_strict = is_strict
        self.is_inter = is_inter
        self.scope = scope
        self.vtype = vtype

    def __format__(self, __format_spec: str) -> str:
        return f"{self.name}: lb={self.lb}, ub={self.ub}, scope={self.scope}"
