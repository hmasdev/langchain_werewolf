from collections import defaultdict
import os
from typing import Generator
from dotenv import load_dotenv
from flaky import flaky
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pytest_mock import MockerFixture
from langchain_werewolf.const import (
    BASE_LANGUAGE,
    DEFAULT_MODEL,
    MODEL_SERVICE_MAP,
)
from langchain_werewolf.enums import ELanguage
from langchain_werewolf.llm_utils import (
    create_chat_model,
    extract_name,
    create_translator_runnable,
)

load_dotenv()


def _int_generator() -> Generator[int, None, None]:
    i = 0
    while True:
        yield i
        i += 1


generate_int = _int_generator().__next__


def test_create_chat_model_with_default(
    mocker: MockerFixture,
) -> None:
    # preparation
    seed = generate_int()
    base_chat_model_mock = mocker.MagicMock()
    mocker.patch('langchain_werewolf.llm_utils._service2cls', defaultdict(lambda: base_chat_model_mock))  # noqa
    # execution
    actual = create_chat_model(seed=seed)
    # assert
    assert actual is create_chat_model(seed=seed)  # check cache
    base_chat_model_mock.assert_called_once_with(model=DEFAULT_MODEL, seed=seed)  # noqa


@pytest.mark.parametrize(
    'llm_name',
    list(MODEL_SERVICE_MAP.keys()),
)
def test_create_chat_model_with_str(
    llm_name: str,
    mocker: MockerFixture,
) -> None:
    # preparation
    seed = generate_int()
    base_chat_model_mock = mocker.MagicMock()
    mocker.patch('langchain_werewolf.llm_utils._service2cls', defaultdict(lambda: base_chat_model_mock))  # noqa
    # execution
    actual = create_chat_model(llm_name, seed=seed)
    # assert
    assert actual is create_chat_model(llm_name, seed=seed)
    base_chat_model_mock.assert_called_once_with(model=llm_name, seed=seed)


def test_create_chat_model_with_base_chat_model(
    mocker: MockerFixture,
) -> None:
    # preparation
    base_chat_model_mock = mocker.MagicMock(spec=ChatOpenAI)
    # execution
    actual = create_chat_model(base_chat_model_mock, seed=generate_int())
    # assert
    assert actual is base_chat_model_mock


def test_create_chat_model_with_unknown_model_name(
    mocker: MockerFixture,
) -> None:
    # preparation
    unknown_model_name = 'unknown_model_name'
    mocker.patch('langchain_werewolf.llm_utils._service2cls', {})
    # assert
    with pytest.raises(ValueError):
        create_chat_model(unknown_model_name)


@pytest.mark.integration
@pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason='OPENAI_API_KEY is not set.',
)
@pytest.mark.parametrize(
    'message, valid_names, context, expected',
    [
        (
            'Call Alice',
            ['Alice', 'Bob', 'Charley'],
            None,
            'Alice',
        ),
        (
            'Call Alice',
            ['Alice', 'Bob', 'Charley'],
            'Who is called?',
            'Alice',
        ),
        (
            'Call Alice. Acutually, not Alice but Charley',
            ['Alice', 'Bob', 'Charley'],
            'Who is called?',
            'Charley',
        )
    ],
)
@flaky(max_runs=3, min_passes=1)
def test_extract_name(
    message: str,
    valid_names: list[str],
    context: str,
    expected: str,
) -> None:
    # TODO: add unit test for extract_name
    # preparation
    chat_model = ChatOpenAI(model='gpt-4o-mini')
    # execution
    actual = extract_name(message, valid_names, context, chat_model=chat_model)
    # assert
    assert actual == expected


def test_create_translator_with_base_language(
    mocker: MockerFixture,
) -> None:
    # preparation
    inputs = ['Hello', 'Goodbye']
    expecteds = ['Hello', 'Goodbye']
    to_language = BASE_LANGUAGE
    chat_model = mocker.MagicMock(spec=BaseChatModel)
    # execution
    translator = create_translator_runnable(to_language, chat_model)
    actuals = translator.batch(inputs)
    # assert
    assert actuals == expecteds


@pytest.mark.integration
@pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason='OPENAI_API_KEY is not set.',
)
@pytest.mark.parametrize(
    'to_language, inputs, expecteds',
    [
        (
            ELanguage.Japanese,
            ['Hello', 'Goodbye'],
            ['こんにちは', 'さようなら'],
        ),
        (
            ELanguage.German,
            ['Hello', 'Goodbye'],
            ['Hallo', 'Auf Wiedersehen'],
        ),
    ]
)
@flaky(max_runs=3, min_passes=1)
def test_create_translator_runnable(
    to_language: ELanguage,
    inputs: list[str],
    expecteds: list[str],
) -> None:
    # TODO: add unit test for create_translator_runnable
    # TODO: add a test with another prompt_template
    # preparation
    chat_model = ChatOpenAI(model='gpt-4o-mini')
    # execution
    translator = create_translator_runnable(to_language, chat_model)
    actuals = translator.batch(inputs)
    # assert
    assert actuals == expecteds
