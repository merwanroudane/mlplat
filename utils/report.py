"""Markdown report generation for projects and capstones (sections 3, 65, 66).

Every exported report carries the platform identity and the package versions
used, so it is self-documenting and reproducible.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import APP_AUTHOR_AR, APP_AUTHOR_EN, APP_NAME_AR, APP_NAME_EN
from utils.validation import package_versions


def _table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(title: str, sections: list[tuple[str, object]], subtitle: str = "") -> str:
    """sections: list of (heading, content) where content is str, DataFrame, dict or list of str."""
    out = [f"# {title}", ""]
    if subtitle:
        out += [subtitle, ""]
    out += [f"*{APP_NAME_AR} · {APP_NAME_EN}*  ", f"*تطوير: {APP_AUTHOR_AR} · Developed by {APP_AUTHOR_EN}*  ",
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*", ""]
    for heading, content in sections:
        out += [f"## {heading}", ""]
        if isinstance(content, pd.DataFrame):
            out.append(_table(content))
        elif isinstance(content, dict):
            out += [f"- **{k}:** {v}" for k, v in content.items()]
        elif isinstance(content, (list, tuple)):
            out += [f"- {item}" for item in content]
        else:
            out.append(str(content))
        out.append("")
    versions = package_versions(("scikit-learn", "numpy", "pandas", "scipy", "streamlit"))
    out += ["## Reproducibility", "", "- " + ", ".join(f"{k} {v}" for k, v in versions.items()), ""]
    return "\n".join(out)
