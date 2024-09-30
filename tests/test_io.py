from collections import defaultdict
from typing import Callable
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.enums import EInputOutputType
from langchain_werewolf.io import (
    create_input_runnable,
    create_output_runnable,
)


@pytest.mark.parametrize(
    'input_func, styler, inputs, expecteds',
    [
        (
            lambda x: x + '!',
            None,
            ['Hello, world', 'Goodbye, world'],
            ['Hello, world!', 'Goodbye, world!'],
        ),
        (
            lambda x: x + '!',
            lambda x: x.upper(),
            ['Hello, world', 'Goodbye, world'],
            ['HELLO, WORLD!', 'GOODBYE, WORLD!'],
        ),
    ],
)
def test_create_input_runnable(
    input_func: Callable[[str], str],
    styler: Callable[[str], str] | None,
    inputs: list[str],
    expecteds: list[str],
) -> None:
    runnable = create_input_runnable(input_func, styler)
    actuals = [runnable.invoke(input_) for input_ in inputs]
    assert actuals == expecteds


@pytest.mark.parametrize(
    'input_func, styler',
    [
        (
            EInputOutputType.standard,
            None,
        ),
        (
            EInputOutputType.click,
            lambda x: x.upper(),
        ),
    ],
)
def test_create_input_runnable_for_einputoutputtype(
    input_func: EInputOutputType,
    styler: Callable[[str], str] | None,
    mocker: MockerFixture,
) -> None:
    # preparation
    inputs = ['Hello, world', 'Goodbye, world']
    expecteds = [styler(s) + '!' if styler else f'{s}!' for s in inputs]
    mocker.patch('langchain_werewolf.io._input_map', defaultdict(lambda: (lambda x: x + '!')))  # noqa
    # run
    runnable = create_input_runnable(input_func, styler)
    actuals = [runnable.invoke(input_) for input_ in inputs]
    # assert
    assert actuals == expecteds


def test_create_input_runnable_invalid_input_func(
    mocker: MockerFixture,
) -> None:
    mocker.patch('langchain_werewolf.io._input_map', {})
    with pytest.raises(ValueError):
        create_input_runnable(EInputOutputType.standard)


@pytest.mark.parametrize(
    'styler, inputs, expected_inputs_to_output_func',
    [
        (
            None,
            ['Hello, world', 'Goodbye, world'],
            ['Hello, world', 'Goodbye, world'],
        ),
        (
            lambda x: x.upper(),
            ['Hello, world', 'Goodbye, world'],
            ['HELLO, WORLD', 'GOODBYE, WORLD'],
        ),
    ],
)
def test_create_output_runnable(
    styler: Callable[[str], str] | None,
    inputs: list[str],
    expected_inputs_to_output_func: list[str],
    mocker: MockerFixture,
) -> None:
    output_func = mocker.MagicMock(spec=Callable[[str], None])
    runnable = create_output_runnable(output_func, styler)
    for input_ in inputs:
        runnable.invoke(input_)
    assert output_func.call_args_list == [
        ((expected_input,),)
        for expected_input in expected_inputs_to_output_func
    ]


@pytest.mark.parametrize(
    'output_func, styler, inputs, expected_inputs_to_output_func',
    [
        (
            EInputOutputType.standard,
            None,
            ['Hello, world', 'Goodbye, world'],
            ['Hello, world', 'Goodbye, world'],
        ),
        (
            EInputOutputType.click,
            lambda x: x.upper(),
            ['Hello, world', 'Goodbye, world'],
            ['HELLO, WORLD', 'GOODBYE, WORLD'],
        ),
    ],
)
def test_create_output_runnable_for_einputoutputtype(
    output_func: EInputOutputType,
    styler: Callable[[str], str] | None,
    inputs: list[str],
    expected_inputs_to_output_func: list[str],
    mocker: MockerFixture,
) -> None:
    # preparation
    output_func_ = mocker.MagicMock(spec=Callable[[str], None])
    mocker.patch('langchain_werewolf.io._output_map', defaultdict(lambda: output_func_))  # noqa
    # run
    runnable = create_output_runnable(output_func, styler)
    for input_ in inputs:
        runnable.invoke(input_)
    # assert
    assert output_func_.call_args_list == [
        ((expected_input,),)
        for expected_input in expected_inputs_to_output_func
    ]


def test_create_output_runnable_invalid_output_func(
    mocker: MockerFixture,
) -> None:
    mocker.patch('langchain_werewolf.io._output_map', {})
    with pytest.raises(ValueError):
        create_output_runnable(EInputOutputType.standard)
