from functools import partial
from typing import Any, Callable
import click
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from .enums import EInputOutputType


_input_map: dict[EInputOutputType, Callable[[str], Any]] = {
    EInputOutputType.standard: input,
    EInputOutputType.click: click.prompt,
}

_output_map: dict[EInputOutputType, Callable[[Any], None]] = {
    EInputOutputType.standard: print,
    EInputOutputType.click: click.echo,
}


def create_input_runnable(
    input_func: Callable[[str], Any] | EInputOutputType = click.prompt,
    styler: Callable[[str], str] | None = None,
    **kwargs,
) -> Runnable[str, str]:
    if isinstance(input_func, EInputOutputType):
        try:
            input_func = _input_map[input_func]
        except KeyError:
            raise ValueError(f'Invalid input_func: {input_func}')
    kwargs = {k: v for k, v in kwargs.items() if k != 'styler'}
    return (
        (RunnableLambda(styler) if styler else RunnablePassthrough())
        | RunnableLambda(partial(input_func, **kwargs))
        | RunnableLambda(str)
    ).with_types(
        input_type=str,
        output_type=str,
    )


def create_output_runnable(
    output_func: Callable[[Any], None] | EInputOutputType = click.echo,
    styler: Callable[[Any], str] | None = None,
    **kwargs,
) -> Runnable[str, None]:
    if isinstance(output_func, EInputOutputType):
        try:
            output_func = _output_map[output_func]
        except KeyError:
            raise ValueError(f'Invalid output_func: {output_func}')
    kwargs = {k: v for k, v in kwargs.items() if k != 'styler'}
    return (
        (RunnableLambda(styler) if styler else RunnablePassthrough())
        | RunnableLambda(partial(output_func, **kwargs))

    ).with_types(
        input_type=str,
        output_type=None,
    )
