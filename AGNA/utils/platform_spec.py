from typing import Any, Dict, List
from utils.base_param import BaseParam


class PlatformSpec(BaseParam):

    name: str
    max_dsp: int
    max_bram: int
    frequency: int
    dbus_width: int
    data_width: int
    alpha: float
    """DSP cost of one MACC operation."""
    beta: List[int]
    """Memory cost of each buffer."""
    gamma_rf: int
    """Capacity of RF."""
    gamma_buf: List[float]
    """Capacity of each buffer."""

    @staticmethod
    def get_param_keys() -> List[str]:
        return [
            "name",
            "max_dsp",
            "max_bram",
            "frequency",
            "dbus_width",
            "data_width",
            "alpha",
            "beta",
            "gamma_rf",
            "gamma_buf",
        ]

    def set_params(self, params: Dict[str, Any]) -> None:
        params.setdefault("alpha", params["data_width"] / 16)
        params.setdefault("beta", [2, 2, 2])
        params.setdefault("gamma_rf", 1024)
        params.setdefault("gamma_buf", [v * 16384 for v in params["beta"]])
        return super().set_params(params)
