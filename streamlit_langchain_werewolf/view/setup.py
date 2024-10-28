from uuid import uuid4
from langchain_werewolf.const import BASE_LANGUAGE, MODEL_SERVICE_MAP
from langchain_werewolf.enums import EChatService, ELanguage, ERole, ESystemOutputType  # noqa
from langchain_werewolf.models.config import Config, PlayerConfig
from langchain_werewolf.utils import consecutive_string_generator
import streamlit as st
from streamlit_langchain_werewolf.const import HUMAN
from streamlit_langchain_werewolf.model import GameSetupModel
from streamlit_langchain_werewolf.session import Session
from streamlit_langchain_werewolf.validator import validate_number_of_players, validate_player_configs  # noqa
from streamlit_langchain_werewolf.view.utils import with_streamlit_placeholder


@with_streamlit_placeholder
def _input_basic_game_settings(
    *,
    disabled: bool = False,
) -> dict[str, int]:

    cols = st.columns(4)
    n_players = cols[0].number_input('Number of Players', min_value=3, max_value=20, value=6, disabled=disabled)  # noqa
    n_werewolves = cols[1].number_input('Number of Werewolves', min_value=1, max_value=5, value=2, disabled=disabled)  # noqa
    n_knights = cols[2].number_input('Number of Knights', min_value=0, max_value=5, value=0, disabled=disabled)  # noqa
    n_fortune_tellers = cols[3].number_input('Number of Fortune Tellers', min_value=0, max_value=5, value=0, disabled=disabled)  # noqa

    validated = validate_number_of_players(n_players, n_werewolves, n_knights, n_fortune_tellers, alert_func=st.error)  # type: ignore # noqa
    if not validated:
        st.stop()

    return dict(
        n_players=n_players,
        n_werewolves=n_werewolves,
        n_knights=n_knights,
        n_fortune_tellers=n_fortune_tellers,
    )


@with_streamlit_placeholder
def _input_player_configs(
    session: Session,
    *,
    n_players: int,
    n_werewolves: int,
    n_knights: int,
    n_fortune_tellers: int,
    disabled: bool = False,
) -> list[PlayerConfig]:
    # input
    default_name_generator = consecutive_string_generator('Player', start=1)
    players = [
        _input_a_player_config(
            default_name=session.game_setting.config.players[i].name,
            default_role=session.game_setting.config.players[i].role,
            default_model=session.game_setting.config.players[i].model,
            default_language=session.game_setting.config.players[i].language,  # noqa
            base_key=f'Player{i}',
            disabled=disabled,
        )
        if session.game_setting and i < len(session.game_setting.config.players) else  # noqa
        _input_a_player_config(
            default_name=default_name_generator.__next__(),
            default_role=None,
            default_model='gpt-4o-mini',
            default_language=BASE_LANGUAGE,
            base_key=f'Player{i}',
            disabled=disabled,
        )
        for i in range(n_players)
    ]
    # validate
    validated = validate_player_configs(
        players,
        n_players=n_players,
        max_nums_for_roles={
            ERole.Villager: n_players - n_werewolves - n_knights - n_fortune_tellers,  # noqa
            ERole.Werewolf: n_werewolves,
            ERole.Knight: n_knights,
            ERole.FortuneTeller: n_fortune_tellers,
        },
        alert_func=st.error,  # type: ignore
    )
    if not validated:
        st.stop()

    # return
    return players


@with_streamlit_placeholder
def _input_a_player_config(
    *,
    default_name: str = 'Player',
    default_role: ERole | None = None,
    default_model: str = 'gpt-4o-mini',
    default_language: ELanguage = BASE_LANGUAGE,
    base_key: str | None = None,
    disabled: bool = False,
) -> PlayerConfig:

    model_options: list[str] = [HUMAN] + [k for k, v in MODEL_SERVICE_MAP.items() if v in {EChatService.OpenAI, EChatService.Google, EChatService.Groq}]  # noqa
    role_options: list[str | None] = [None]+[e.value for e in ERole]
    language_options: list[str] = [e.value for e in ELanguage]

    default_role_idx: int = role_options.index(default_role.value if default_role else None)  # noqa
    default_model_idx: int = model_options.index(default_model)
    default_language_idx: int = language_options.index(default_language.value)

    name_key = f'name-{base_key}' if base_key else str(uuid4())
    role_key = f'role-{base_key}' if base_key else str(uuid4())
    model_key = f'model-{base_key}' if base_key else str(uuid4())
    language_key = f'language-{base_key}' if base_key else str(uuid4())

    cols = st.columns(4)
    name = cols[0].text_input('Player Name', value=default_name, key=name_key, disabled=disabled)  # noqa
    role = cols[1].selectbox('Role of the player', options=role_options, index=default_role_idx, key=role_key, disabled=disabled)  # noqa
    model = cols[2].selectbox('Model', options=model_options, index=default_model_idx, key=model_key, disabled=disabled)  # noqa
    language = cols[3].selectbox('Language', options=language_options, index=default_language_idx, key=language_key, disabled=disabled)  # noqa

    erole = ERole(role) if role is not None else role
    elanguage = ELanguage(language)

    return PlayerConfig(
        name=name,
        role=erole,
        model=model,
        language=elanguage,
    )


@with_streamlit_placeholder
def _input_global_llm_settings(
    *,
    disabled: bool = False,
    default_model: str = 'gpt-4o-mini',
) -> dict[str, str | int]:
    cols = st.columns(2)
    models = list(MODEL_SERVICE_MAP.keys())
    model = cols[0].selectbox('Model', options=models, index=models.index(default_model), disabled=disabled)  # noqa
    seed = cols[1].number_input('Seed', min_value=-1, value=-1, step=1, disabled=disabled)  # noqa
    return dict(
        model=model,
        seed=seed,
    )


@with_streamlit_placeholder
def _input_system_output_config(
    *,
    player_names: list[str],
    default_output_level: ESystemOutputType = ESystemOutputType.all,
    default_language: ELanguage = BASE_LANGUAGE,
) -> dict[str, ELanguage | ESystemOutputType | str]:
    cols = st.columns(2)

    language_options: list[str] = [e.value for e in ELanguage]
    default_language_idx: int = language_options.index(default_language.value)
    language = ELanguage(cols[0].selectbox('System Language', options=language_options, index=default_language_idx))  # noqa

    output_level_options: list[str] = [e.value for e in ESystemOutputType] + player_names  # noqa
    default_output_level_idx: int = output_level_options.index(default_output_level.value)  # noqa
    output_level_raw = cols[1].selectbox('System Output Level', options=output_level_options, index=default_output_level_idx)  # noqa
    output_level = output_level_raw if output_level_raw in player_names else ESystemOutputType(output_level_raw)  # noqa

    return dict(
        system_language=language,
        system_output_level=output_level,
    )


@with_streamlit_placeholder
def setup_view(
    session: Session,
    *,
    disabled: bool = False,
) -> GameSetupModel:

    basic_game_settings = _input_basic_game_settings(disabled=disabled)  # noqa
    players = _input_player_configs(
        session,
        **basic_game_settings,
        placeholder=st.expander(label='Detailed Player Settings', expanded=True),  # noqa
        disabled=disabled,
    )
    config = Config(players=players)
    global_llm_setting = _input_global_llm_settings(
        disabled=disabled,
        placeholder=st.expander(label='Global LLM Settings', expanded=False),  # noqa
    )

    if all([p.model != HUMAN for p in players]):
        system_output_config = _input_system_output_config(
            player_names=[p.name for p in players],
            placeholder=st.expander(label='System Output Config', expanded=False),  # noqa
        )
    else:
        system_output_config = {}

    return GameSetupModel(
        **basic_game_settings,  # type: ignore
        **global_llm_setting,  # type: ignore
        **system_output_config,  # type: ignore
        config=config,
    )
