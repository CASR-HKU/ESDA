from typing_extensions import Self
from typing import Any, Dict, List, Iterator, Set
import json
import logging

from utils.my_logger import get_logger


class BaseParamJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseParam):
            return dict(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class BaseParam:
    """Base class for other param/spec classes."""

    logger: logging.Logger

    @staticmethod
    def get_param_keys() -> List[str]:
        raise NotImplementedError(f"Overriding is requried.")

    def set_params(self, params: Dict[str, Any]) -> None:
        for k in self.get_param_keys():
            try:
                v = params[k]
                setattr(self, k, v)
            except KeyError as e:
                self.logger.exception(e)
                raise

    def get_params(self) -> List[Any]:
        return [getattr(self, k) for k in self.get_param_keys()]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize BaseParam.

        Arguments:
            params: A dictionary of parameters.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.set_params(kwargs)

    # def __init__(self, params: Dict[str, Any]) -> None:
    #     """Initialize BaseParam.

    #     Arguments:
    #         params: A dictionary of parameters.
    #     """
    #     self.logger = get_logger(self.__class__.__name__)
    #     self.set_params(params)

    # def __getitem__(self, __key: str, __default: Any = None) -> Any:
    #     return getattr(self, __key)

    def __iter__(self) -> Iterator:
        """For serialization purpose."""
        for k in self.get_param_keys():
            yield k, getattr(self, k)

    def __format__(self, __format_spec: str) -> str:
        if __format_spec == "":
            return str(self)
        else:
            raise ValueError(f"Unknown format_spec: {__format_spec}")

    def save(self, path: str) -> None:
        """Save to a JSON file."""
        with open(path, "w") as f:
            json.dump(self, f, indent=4, cls=BaseParamJSONEncoder)
        self.logger.debug(f"Write to {path}")

    @classmethod
    def load(cls, path: str) -> Self:
        """Load from a JSON file."""
        try:
            with open(path, "r") as f:
                params = json.load(f)
            return cls(**params)
        except FileNotFoundError as e:
            raise

    def summary(self) -> None:
        self.logger.info(f"Summary: {self}")
