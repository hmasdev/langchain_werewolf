from copy import deepcopy
from functools import partial, wraps
import random
import time
from typing import (
    Callable,
    Generator,
    Iterable,
    ParamSpec,
    Sized,
    TypeVar,
    overload,
)

from pydantic import BaseModel
import pydantic_core

P = ParamSpec('P')
T = TypeVar('T')
PydanticBaseModel = TypeVar('PydanticBaseModel', bound=BaseModel)


def assert_not_empty_deco(func: Callable[..., Sized]) -> Callable[..., Sized]:
    """A decorator to assert that the input is not empty

    Args:
        func (Callable[..., Sized]): the function to be decorated

    Returns:
        Callable[..., Sized]: the decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Sized:
        result = func(*args, **kwargs)
        if len(result) == 0:
            raise ValueError('The input is empty.')
        return result
    return wrapper


def consecutive_string_generator(
    prefix: str,
    start: int = 0,
    step: int = 1,
) -> Generator[str, None, None]:
    """a generator to generate strings with a prefix and a number.

    Args:
        prefix (str): prefix of the string
        start (int, optional): start number. Defaults to 0.
        step (int, optional): step of the number. Defaults to 1.

    Yields:
        Generator[str, None, None]: the generated string
            format: f'{prefix}{number}'
    """
    idx = start
    while True:
        yield f'{prefix}{idx}'
        idx += step


def random_permutated_infinite_generator(
    objects: Iterable[T],
) -> Generator[T, None, None]:
    """a generator to generate random permutated objects infinitely

    Args:
        objects (Iterable[T]): the objects to be permutated

    Yields:
        Generator[T, None, None]: infinite generator with random permutated objects
    """  # noqa
    lst_objects = deepcopy(list(objects))
    while True:
        random.shuffle(lst_objects)
        for obj in lst_objects:
            yield obj


def remove_none_values(d: dict[str, T | None]) -> dict[str, T]:
    """Remove the None values from the dictionary

    Args:
        d (dict[str, T  |  None]): the dictionary from which to remove the None values

    Returns:
        dict[str, T]: the dictionary without the None values
    """  # noqa
    return {k: v for k, v in d.items() if v is not None}


def load_json(
    Model: type[PydanticBaseModel],
    path: str,
) -> PydanticBaseModel:
    """Load a pydantic model from a json file

    Args:
        Model (type[PydanticBaseModel]): the pydantic model
        path (str): the path to the file

    Raises:
        FileNotFoundError: the file is not found
        pydantic.ValidationError: the loaded json is invalid

    Returns:
        PydanticBaseModel: the loaded pydantic model
    """  # noqa
    with open(path, 'r') as f:
        contents = f.read()
    return Model.model_validate(
        pydantic_core.from_json(contents, allow_partial=True),
    )


@overload
def delay_deco(
    func: Callable[P, T],
    seconds: float = 1,
) -> Callable[P, T]:
    """A decorator to delay the execution of a function

    Args:
        func (Callable[P, T]): the function to be decorated
        seconds (float, optional): the delay in seconds. Defaults to 1.
    Returns:
        Callable[P, T]: the decorated function
    """
    ...


@overload
def delay_deco(
    func: None = None,
    seconds: float = 1,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """A decorator to delay the execution of a function
    Args:
        func (None, optional): the function to be decorated. Defaults to None.
        seconds (float, optional): the delay in seconds. Defaults to 1.
    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: a decorator
    """
    ...


def delay_deco(
    func: Callable[P, T] | None = None,
    seconds: float = 1,
) -> Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T]:
    """A decorator to delay the execution of a function

    Args:
        func (Callable[P, T] | None, optional):
            the function to be decorated. Defaults to None.
        seconds (float, optional):
            the delay in seconds. Defaults to 1.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T]:
            the decorated function

    Notes:
        - If `func` is None, a decorator is returned.
          Otherwise, the decorated function is returned.
    """

    if func is None:
        return partial(delay_deco, seconds=seconds)

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        time.sleep(seconds)
        return func(*args, **kwargs)
    return wrapper
