"""Content hygiene: no placeholders, no unsafe execution, identity present, light theme, internal links valid."""

import os
import re

from core.curriculum import MODULES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_FILES = [os.path.join(dp, f) for d in ("app_pages", "components", "core", "utils", "content")
            for dp, _, fs in os.walk(os.path.join(ROOT, d)) for f in fs if f.endswith(".py")]
BANNED = [r"\bTODO\b", r"Coming soon", r"[Ll]orem ipsum", r"__STUB__", r"\bFIXME\b", "قيد " + "الإنشاء", "سيتم " + "إضافته"]


def test_no_placeholders():
    for path in PY_FILES:
        text = open(path, encoding="utf-8").read()
        for pat in BANNED:
            assert not re.search(pat, text), (path, pat)


def test_no_arbitrary_code_execution():
    for path in PY_FILES:
        text = open(path, encoding="utf-8").read()
        assert not re.search(r"(?<![\w.])(eval|exec)\s*\(", text), path
        assert not re.search(r"import subprocess|subprocess\.", text), path
        assert "pickle.load" not in text and "joblib.load" not in text, path


def test_pages_are_not_empty():
    for m in MODULES:
        text = open(os.path.join(ROOT, m.file), encoding="utf-8").read()
        assert len(text.splitlines()) >= 15, m.file
        assert "footer" in text, m.file  # identity footer on every page


def test_identity_is_central():
    cfg = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert 'APP_AUTHOR_EN = "Dr. Marwan Roudane"' in cfg
    assert 'DATA_SCIENCE_PLATFORM_URL = "https://dsplat.streamlit.app/"' in cfg
    author_ar = "مروان " + "رودان"
    for path in PY_FILES:
        assert author_ar not in open(path, encoding="utf-8").read(), path  # the author name comes from config only


def test_light_theme_only():
    toml = open(os.path.join(ROOT, ".streamlit", "config.toml"), encoding="utf-8").read()
    assert 'base = "light"' in toml
    assert 'backgroundColor = "#FCFCFF"' in toml and 'primaryColor = "#1971C2"' in toml
    assert 'base = "dark"' not in toml


def test_internal_page_links_resolve():
    ids = {m.id for m in MODULES}
    for path in PY_FILES:
        text = open(path, encoding="utf-8").read()
        for mid in re.findall(r"(?:page_link|get_module)\(\s*\"([a-z_0-9]+)\"", text):
            assert mid in ids, (path, mid)
        for mid in re.findall(r"page_(?:header|footer)\(\s*\"([a-z_0-9]+)\"", text):
            assert mid in ids, (path, mid)
