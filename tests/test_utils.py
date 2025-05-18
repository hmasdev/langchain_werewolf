from typing import Callable, TypeVar
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.utils import (
    assert_not_empty_deco,
    consecutive_string_generator,
    random_permutated_infinite_generator,
    remove_none_values,
)

T = TypeVar('T')


def test_assert_not_empty_deco():
    @assert_not_empty_deco
    def test_func(a: list[int]) -> list[int]:
        return a

    assert test_func([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValueError):
        test_func([])


@pytest.mark.parametrize(
    'prefix, start, step, expected',
    [
        ('a', 0, 1, ['a0', 'a1', 'a2', 'a3', 'a4']),
        ('b', 1, 2, ['b1', 'b3', 'b5', 'b7', 'b9']),
    ],
)
def test_consecutive_string_generator(
    prefix: str,
    start: int,
    step: int,
    expected: list[str],
):
    generator = consecutive_string_generator(prefix, start, step)
    assert [next(generator) for _ in range(5)] == expected


@pytest.mark.parametrize(
    'n_cycle,shuffle,objects,expected',
    [
        (
            5,
            lambda x: x,
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]*5,
        ),
        (
            3,
            list.reverse,
            ['a', 'b', 'c', 'd', 'e'],
            ['e', 'd', 'c', 'b', 'a', 'a', 'b', 'c', 'd', 'e', 'e', 'd', 'c', 'b', 'a'],  # noqa
        ),
    ],
)
def test_random_permutated_infinite_generator(
    n_cycle: int,
    shuffle: Callable[[list[T]], None],
    objects: list[T],
    expected: list[T],
    mocker: MockerFixture,
):
    mocker.patch('random.shuffle', shuffle)
    generator = random_permutated_infinite_generator(objects)
    assert [next(generator) for _ in range(len(objects) * n_cycle)] == expected  # noqa


@pytest.mark.parametrize(
    'd, expected',
    [
        (
            {'a': 1, 'b': 2, 'c': None},
            {'a': 1, 'b': 2},
        ),
        (
            {'a': 1, 'b': 2, 'c': None, 'd': None},
            {'a': 1, 'b': 2},
        ),
        (
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 1, 'b': 2, 'c': 3},
        ),
    ],
)
def test_remove_none_values(
    d: dict[str, T],
    expected: dict[str, T],
):
    assert remove_none_values(d) == expected  # type: ignore
