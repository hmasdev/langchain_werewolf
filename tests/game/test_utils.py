from datetime import datetime as dt
from typing import Callable
from langgraph.graph import Graph, START, END, StateGraph
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.models.state import MsgModel, StateModel
from langchain_werewolf.game.utils import (
    add_echo_node,
    create_message_history_prompt,
)


@pytest.mark.parametrize(
    'formatter',
    [
        MsgModel.format,
        str,
    ],
)
def test_create_message_history_prompt(
    formatter: Callable[[MsgModel], str],
) -> None:
    # preparation
    messages = [
        MsgModel(
            name='sender1',
            timestamp=dt.now(),
            message='content1',
            participants=frozenset(['sender1', 'sender2']),
        ),
        MsgModel(
            name='sender2',
            timestamp=dt.now(),
            message='content2',
            participants=frozenset(['sender1', 'sender2']),
        ),
    ]
    expected = [formatter(msg) for msg in messages]

    # execution
    actual = create_message_history_prompt(messages, formatter)
    # assert
    assert actual == '\n'.join(expected)


def test_add_echo_node_with_echo_being_none() -> None:
    # preparation
    graph = Graph()
    # execution
    graph_with_echo = add_echo_node(graph, 'dummy', echo=None)
    # assert
    assert graph_with_echo is graph


@pytest.mark.parametrize(
    'node',
    [
        '',
        [],
    ],
)
def test_add_echo_node_with_node_empty(
    node: str | list[str],
) -> None:
    # preparation
    graph = Graph()
    # execution
    graph_with_echo = add_echo_node(graph, node, echo=lambda x: print(x))
    # assert
    assert graph_with_echo is graph


@pytest.mark.parametrize(
    'node',
    [
        'node1',
        ['node1', 'node2']
    ],
)
def test_add_echo_node_for_callable_echo(
    node: str | list[str],
    mocker: MockerFixture,
) -> None:
    # preparation
    class Echo:
        def echo(self, state: StateModel) -> None:
            print(state)
    echo = Echo()
    nodes = [node] if isinstance(node, str) else node
    graph = StateGraph(StateModel)
    for n in nodes:
        graph.add_node(n, RunnablePassthrough())
    graph.add_edge(START, nodes[0])
    graph.add_edge(nodes[-1], END)
    if len(nodes) > 1:
        for n1, n2 in zip(nodes, nodes[1:]):
            graph.add_edge(n1, n2)
    echo_spy = mocker.spy(echo, 'echo')
    # execution
    graph_with_echo = add_echo_node(graph, node, echo=echo.echo).compile()
    graph_with_echo.invoke(StateModel(alive_players_names=[]))
    # assert
    assert echo_spy.call_count == len(nodes)


@pytest.mark.parametrize(
    'node',
    [
        'node1',
        ['node1', 'node2']
    ],
)
def test_add_echo_node_for_runnable_echo(
    node: str | list[str],
    mocker: MockerFixture,
) -> None:
    # preparation
    echo = RunnableLambda(lambda x: print(x))
    node = 'node1'
    graph = StateGraph(StateModel)
    graph.add_node(node, RunnablePassthrough())
    graph.add_edge(START, node)
    graph.add_edge(node, END)
    echo_spy = mocker.spy(echo, 'invoke')
    # execution
    graph_with_echo = add_echo_node(graph, node, echo=echo).compile()
    graph_with_echo.invoke(StateModel(alive_players_names=[]))
    # assert
    assert echo_spy.call_count == 1


def test_add_echo_node_with_node_not_found() -> None:
    # preparation
    graph = StateGraph(StateModel)
    graph.add_node('node1', RunnablePassthrough())
    graph.add_edge(START, 'node1')
    graph.add_edge('node1', END)
    # execution
    graph_with_echo = add_echo_node(graph, ['node1', 'node2'], echo=lambda x: print(x)).compile()  # noqa
    graph_with_echo.invoke(StateModel(alive_players_names=[]))
