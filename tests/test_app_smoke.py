"""Streamlit pages must render without exceptions even with no dataset/model (empty states)."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import config

PAGES = ["app.py"] + [f"pages/{p.name}" for p in sorted((config.BASE_DIR / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, tmp_config):
    at = AppTest.from_file(str(config.BASE_DIR / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
