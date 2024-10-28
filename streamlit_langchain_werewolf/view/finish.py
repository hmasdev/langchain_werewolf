from logging import Logger, getLogger
import streamlit as st
from streamlit_langchain_werewolf.session import Session
from streamlit_langchain_werewolf.view.utils import with_streamlit_placeholder


@with_streamlit_placeholder
def finish_view(
    session: Session,
    logger: Logger = getLogger(__name__),
) -> None:
    state = session.result
    if state is None or state.result is None:
        logger.error('No result')
        st.error('No result')
    else:
        logger.info('Game has been finished')
        st.info(state.result.value)
