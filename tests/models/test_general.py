from typing import Generator
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.models.general import (
    IdentifiedModel,
    PartialFrozenModel,
    constant_reducer,
    overwrite_reducer,
    reduce_dict,
    reduce_list,
    _generate_unique_string,
)


@pytest.mark.parametrize(
    'value',
    [
        'contents',
        2,
    ],
)
def test_IdentifiedModel(value: str | int) -> None:
    # preparation
    value = 'contents'
    # execution
    actual1 = IdentifiedModel(value=value)
    actual2 = IdentifiedModel(value=value)
    # assert
    assert actual1.id
    assert actual2.id
    assert actual1.id != actual2.id
    assert actual1.value == value
    assert actual2.value == value


@pytest.mark.parametrize(
    'value, new_value',
    [
        ('contents', 'new contents'),
        (2, 3),
    ],
)
def test_PartialFrozenModel(
    value: str | int,
    new_value: str | int,
) -> None:
    # preparation
    class TestModel(PartialFrozenModel):
        frozen_fields: set[str] = {'value'}
        value: str | int
    # execution
    actual = TestModel(value=value)
    # assert
    assert actual.value == value
    with pytest.raises(TypeError):
        actual.value = new_value
    assert actual.value == value


def test_overwrite_reducer() -> None:
    # preparation
    old = 'old'
    new = 'new'
    # execution
    actual = overwrite_reducer(old, new)
    # assert
    assert actual == new


def test_constant_reducer() -> None:
    # preparation
    old = 'old'
    new = 'new'
    # execution
    actual = constant_reducer(old, new)
    # assert
    assert actual == old


def test__generate_unique_string() -> None:
    # execution
    actual1 = _generate_unique_string()
    actual2 = _generate_unique_string()
    # assert
    assert actual1 != actual2


@pytest.mark.parametrize(
    'old, new, expected',
    [
        ({'old': 'old'}, {'new': 'new'}, {'old': 'old', 'new': 'new'}),
        ({'old': 'old'}, None, {'old': 'old'}),
        (None, {'new': 'new'}, {'new': 'new'}),
        (None, None, {}),
    ],
)
def test_reduce_dict(
    old: dict[str, str] | None,
    new: dict[str, str] | None,
    expected: dict[str, str],
) -> None:
    # execution
    actual = reduce_dict(old, new)
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'old, new, id_generator, expected',
    [
        (
            # Case:
            # - new don't have contents whose ID is same as old's
            # - old and new are lists of not-IdentifierModel
            ['a', 'b'],
            ['c', 'd'],
            (str(i) for i in range(100000)),
            [
                IdentifiedModel[str](id='0', value='a'),
                IdentifiedModel[str](id='1', value='b'),
                IdentifiedModel[str](id='2', value='c'),
                IdentifiedModel[str](id='3', value='d'),
            ],
        ),
        (
            # Case:
            # - new don't have contents whose ID is same as old's
            # - new is a list of not-IdentifierModel
            [
                IdentifiedModel[str](id='a', value='a'),
                IdentifiedModel[str](id='b', value='b'),
            ],
            ['c', 'd'],
            (str(i) for i in range(100000)),
            [
                IdentifiedModel[str](id='a', value='a'),
                IdentifiedModel[str](id='b', value='b'),
                IdentifiedModel[str](id='0', value='c'),
                IdentifiedModel[str](id='1', value='d'),
            ],
        ),
        (
            # Case:
            # - new have some contents whose ID is same as old's
            # - somd of new are not IdentifierModel
            [
                IdentifiedModel[str](id='a', value='a'),
                IdentifiedModel[str](id='b', value='b'),
            ],
            [
                IdentifiedModel[str](id='b', value='b'),
                'd',
            ],
            (str(i) for i in range(100000)),
            [
                IdentifiedModel[str](id='a', value='a'),
                IdentifiedModel[str](id='b', value='b'),
                IdentifiedModel[str](id='0', value='d'),
            ],
        ),
        (
            # Case
            # - new have some contents whose ID is same as old's
            # - all elements of new are instances of IdentifierModel
            [
                IdentifiedModel[str](id='a', value='a'),
                IdentifiedModel[str](id='b', value='b'),
            ],
            [
                IdentifiedModel[str](id='b', value='b'),
                IdentifiedModel[str](id='d', value='d'),
            ],
            (str(i) for i in range(100000)),
            [
                IdentifiedModel[str](id='a', value='a'),
                IdentifiedModel[str](id='b', value='b'),
                IdentifiedModel[str](id='d', value='d'),
            ],
        )
    ],
)
def test_reduce_list(
    old: list[str] | None,
    new: list[str] | None,
    id_generator: Generator[str, None, None],
    expected: list[dict[str, str]],
    mocker: MockerFixture,
) -> None:
    # preparation
    mocker.patch('langchain_werewolf.models.general.uuid.uuid4', id_generator.__next__)  # noqa
    # execution
    actual: list = reduce_list(old, new)  # type: ignore
    # assert
    assert actual == expected
