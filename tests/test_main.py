import os
from flaky import flaky
from dotenv import load_dotenv
import pytest
from langchain_werewolf.enums import (
    EInputOutputType,
    ELanguage,
    ESystemOutputType,
)
from langchain_werewolf.game_players.player_roles import (
    FortuneTeller,
    Knight,
    Werewolf,
)
from langchain_werewolf.main import main
from langchain_werewolf.models.config import Config, GeneralConfig

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason='OPENAI_API_KEY is not set.',
)
@flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize(
    'config',
    [
        Config(
            general=GeneralConfig(
                n_players=4,
                n_players_by_role={
                    Werewolf.role: 1,
                    Knight.role: 1,
                    FortuneTeller.role: 1,
                },
                seed=-1,
            ),
        ),
        Config(
            general=GeneralConfig(
                n_players=4,
                n_players_by_role={
                    Werewolf.role: 1,
                    Knight.role: 1,
                    FortuneTeller.role: 1,
                },
                seed=-1,
                system_language=ELanguage.Japanese,
            ),
        ),
        Config(
            general=GeneralConfig(
                n_players=4,
                n_players_by_role={
                    Werewolf.role: 1,
                    Knight.role: 1,
                    FortuneTeller.role: 1,
                },
                seed=-1,
                system_language=ELanguage.German,
            ),
        ),
    ]
)
def test_main_integration(config: Config) -> None:
    state = main(
        n_players=4,
        n_players_by_role={
            Werewolf.role: 1,
            Knight.role: 1,
            FortuneTeller.role: 1,
        },
        output='',
        system_output_level=ESystemOutputType.off,
        system_output_interface=EInputOutputType.standard,
        seed=-1,
        config=config,
    )
    assert state.result is not None
