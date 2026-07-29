from collections.abc import Iterator
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def safe_tmp_path() -> Iterator[Path]:
    """Temporário único, sem usar o diretório global gerenciado pelo pytest."""
    with tempfile.TemporaryDirectory(prefix="network_momentum_test_") as directory:
        yield Path(directory)

