from typing import Any
from abc import ABC, abstractmethod


class ServiceInterface(ABC):
    """An abstract base class for defining service interfaces."""

    def __init__(self, name: str = "undefined"):
        self.name = name

    @abstractmethod
    def inference(self, payload: Any, *args, **kwargs) -> Any:
        """
        Performs inference using the provided payload.

        Args:
            payload: The data to be used for inference.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The inference result.
        """

        raise NotImplementedError
