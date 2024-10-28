import streamlit as st
from streamlit_langchain_werewolf.view.utils import with_streamlit_placeholder


@with_streamlit_placeholder
def header_view():
    st.title('⛓🐺⛓Langchain Werewolf⛓🐺⛓')
