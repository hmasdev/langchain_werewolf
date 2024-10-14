import os
from flaky import flaky
from dotenv import load_dotenv
import pytest
from langchain_werewolf.enums import EInputOutputType, ESystemOutputType
from langchain_werewolf.main import main

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason='OPENAI_API_KEY is not set.',
)
@flaky(max_runs=2, min_passes=1)
def test_main_integration() -> None:
    main(
        n_players=4,
        n_werewolves=1,
        n_knights=1,
        n_fortune_tellers=1,
        output='',
        system_output_level=ESystemOutputType.off,
        system_output_interface=EInputOutputType.standard,
        seed=-1,
    )
