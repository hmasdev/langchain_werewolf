from typing import Callable, TypeVar
from streamlit.delta_generator import DeltaGenerator

T = TypeVar('T')


def with_streamlit_placeholder(
    func: Callable[..., T],
) -> Callable[..., T]:

    def _with_kwarg_for_streamlit_placeholder_wrapper(
        *args,
        placeholder: DeltaGenerator | None = None,
        **kwargs,
    ) -> T:
        # Case: placeholder given
        if placeholder is not None:
            with placeholder.container():
                kwargs['placeholder'] = None
                return _with_kwarg_for_streamlit_placeholder_wrapper(*args, **kwargs)  # noqa
        return func(*args, **kwargs)

    return _with_kwarg_for_streamlit_placeholder_wrapper


def with_streamlit_expander(
    func: Callable[..., T],
) -> Callable[..., T]:

    def _with_kwarg_for_streamlit_expander_wrapper(
        *args,
        expander: DeltaGenerator | None = None,
        **kwargs,
    ) -> T:
        # Case: expander given
        if expander is not None:
            with expander:
                kwargs['expander'] = None
                return _with_kwarg_for_streamlit_expander_wrapper(*args, **kwargs)  # noqa
        return func(*args, **kwargs)

    return _with_kwarg_for_streamlit_expander_wrapper
