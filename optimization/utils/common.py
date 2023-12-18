from functools import reduce
from typing import Any, Iterable


def csv_row(iterable: Iterable[Any]) -> str:
    str_list = [str(i) for i in iterable]
    return ",".join(s if "," not in s else f'"{s}"' for s in str_list)


def my_sum(iterable: Iterable[Any]) -> Any:
    return reduce(lambda x, y: x + y, iterable)


def my_prod(iterable: Iterable[Any]) -> Any:
    return reduce(lambda x, y: x * y, iterable)
