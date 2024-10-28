from typing import Iterable
from streamlit_langchain_werewolf.model import StremlitMessageModel


def messasges_require_response(
    messages: StremlitMessageModel | Iterable[StremlitMessageModel],
) -> bool:
    messages = [messages] if isinstance(messages, StremlitMessageModel) else messages  # noqa
    return any([m for m in messages if m.response_required])
