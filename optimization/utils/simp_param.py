from typing import List
from utils.base_param import BaseParam


class SimpParam(BaseParam):
    a: int
    b: int
    obj: int

    @staticmethod
    def get_param_keys() -> List[str]:
        return ["a", "b", "obj"]

    def __format__(self, __format_spec: str) -> str:
        if __format_spec == "":
            return f"a={self.a}, b={self.b}, obj={self.obj}"
        else:
            return super().__format__(__format_spec)
