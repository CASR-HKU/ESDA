from typing import List
from utils.base_param import BaseParam

class ArchParam(BaseParam):

    A: List[int]
    pe_num: int

    @staticmethod
    def get_param_keys() -> List[str]:
        return ["A", "pe_num"]

    def __format__(self, __format_spec: str) -> str:
        if __format_spec == "":
            return f"A={self.A}, pe_num={self.pe_num}"
        else:
            return super().__format__(__format_spec)
        