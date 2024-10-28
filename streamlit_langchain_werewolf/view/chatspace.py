from typing import Iterable
import streamlit as st
from langchain_werewolf.models.state import MsgModel
from streamlit_langchain_werewolf.model import StremlitMessageModel
from streamlit_langchain_werewolf.view.utils import with_streamlit_placeholder


@with_streamlit_placeholder
def messages_view(
    contents: StremlitMessageModel | Iterable[StremlitMessageModel],
) -> None:
    contents = [contents] if isinstance(contents, StremlitMessageModel) else contents  # noqa
    for content in contents:
        try:
            msg = MsgModel.model_validate_json(content.message)
        except Exception:
            st.write(content.message)
            continue
        timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        with st.chat_message(name=msg.name):
            st.write(f'{msg.name} ({timestamp})')
            st.write('participants: '+', '.join(msg.participants))
            st.write('-----')
            with st.container(border=True):
                st.markdown(msg.message.replace('\n', '\n\n'))
            st.write('-----')


@with_streamlit_placeholder
def input_chat_message_view(
    *,
    disabled: bool = False,
) -> str | None:
    return st.chat_input(placeholder='Input Your Message', disabled=disabled)
