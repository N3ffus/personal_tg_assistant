import os

import pytest

from src.config import Settings

# graphiti-core calls load_dotenv() when it is imported, which leaks the
# developer's .env into os.environ. Keep it out of the settings under test.
APPLICATION_ENV_VARS = tuple(name.upper() for name in Settings.model_fields)


@pytest.fixture(autouse=True)
def isolated_application_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in APPLICATION_ENV_VARS:
        if name in os.environ:
            monkeypatch.delenv(name)
