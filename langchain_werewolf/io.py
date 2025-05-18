from functools import partial
from typing import Any, Callable
import click
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from .enums import EInputOutputType
from .utils import delay_deco


def attach_prefix_to_prompt(
    input_func: Callable[[str], Any],
    prefix: str = "",
) -> Callable[[str], Any]:

    def wrapped_input_func(prompt: str, *args, **kwargs) -> Any:
        prompt = f"{prefix} {prompt}"
        return input_func(prompt, *args, **kwargs)

    return wrapped_input_func


_input_map: dict[EInputOutputType, Callable[[str], Any]] = {
    EInputOutputType.none: lambda _: None,
    EInputOutputType.standard: delay_deco(attach_prefix_to_prompt(input), seconds=2),  # noqa
    EInputOutputType.click: delay_deco(attach_prefix_to_prompt(click.prompt), seconds=2),  # noqa
    # FIXME: fix the above `delay_deco`. This is a patch to avoid the conflict between the output and the input prompt.  # noqa
}

_output_map: dict[EInputOutputType, Callable[[Any], None]] = {
    EInputOutputType.none: lambda _: None,
    EInputOutputType.standard: print,
    EInputOutputType.click: click.echo,
}


def create_input_runnable(
    input_func: Callable[[str], Any] | EInputOutputType = click.prompt,
    styler: Callable[[str], str] | None = None,
) -> Runnable[str, str]:
    if isinstance(input_func, EInputOutputType):
        try:
            input_func = _input_map[input_func]
        except KeyError:
            raise ValueError(f'Invalid input_func: {input_func}')
    return (
        (RunnableLambda(styler) if styler else RunnablePassthrough())
        | RunnableLambda(input_func)
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
