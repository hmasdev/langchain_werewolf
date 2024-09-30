from logging import getLogger, Logger
from typing import Callable, Iterable
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import Graph, END
from ..models.state import (
    MsgModel,
    StateModel,
    create_dict_without_state_updated,
)


def create_message_history_prompt(
    messages: list[MsgModel],
    formatter: Callable[[MsgModel], str] = MsgModel.format,
) -> str:
    return '\n'.join([
        formatter(message)
        for message in messages
    ])


def add_echo_node(
    workflow: Graph,
    node: str | Iterable[str],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
    echo_node_name: str = '_echo_',
    next_node: str = END,
    logger: Logger = getLogger(__name__),
) -> Graph:
    if echo is None:
        return workflow
    if not node:
        return workflow
    # initialize
    nodes = [node] if isinstance(node, str) else node
    # add output node
    if not isinstance(echo, Runnable):
        echo = RunnableLambda(echo)
    workflow.add_node(
        echo_node_name,
        echo | RunnableLambda(create_dict_without_state_updated),
    )
    # add edges
    for node in nodes:
        if node not in workflow.nodes:
            logger.warning(f'Node {node} not found in the workflow')
            continue
        workflow.add_edge(node, echo_node_name)

    workflow.add_edge(echo_node_name, next_node)
    return workflow
