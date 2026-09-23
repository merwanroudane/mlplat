"""Central configuration: identity, naming and global constants.

Every page reads the author name, platform name and compute limits from here,
so they are never hard-coded anywhere else.
"""

from __future__ import annotations

APP_AUTHOR_AR = "الدكتور مروان رودان"
APP_AUTHOR_EN = "Dr. Marwan Roudane"

APP_NAME_AR = "أكاديمية تعلّم الآلة التفاعلية"
APP_NAME_EN = "Machine Learning Interactive Academy"
APP_SUBTITLE_AR = (
    "منصة أكاديمية تفاعلية لتعلّم الآلة: من جاهزية البيانات إلى الخوارزميات والتحسين والتقييم "
    "والتفسير والإنتاج، مع مسار مستقل ومتعمق في Double Machine Learning"
)
APP_VERSION = "1.1.0"
APP_SLUG = "MLplat"

DATA_SCIENCE_PLATFORM_URL = "https://dsplat.streamlit.app/"
DATA_SCIENCE_REPO_URL = "https://github.com/merwanroudane/DSplat"

# Date on which versions, defaults and APIs were verified (see docs/research_notes.md)
VERIFIED_ON = "2026-09-23"

# Global random seed used by synthetic datasets, splits and simulations
RANDOM_SEED = 42

# ------------------------------------------------------------ compute limits
# Public deployment (Streamlit Community Cloud) has ~1 CPU and limited RAM:
# every lab clamps its work to these bounds (section 82 of the brief).
MAX_SAMPLES_LAB = 3_000        # rows used by any interactive training lab
MAX_TREES_LAB = 300            # trees in forests / boosting rounds in labs
MAX_OPTUNA_TRIALS = 60         # trials in the HPO lab
MAX_MC_REPS = 200              # Monte Carlo replications in simulation labs
MAX_PLOT_POINTS = 5_000        # points in a scatter before sampling
MAX_UPLOAD_MB = 20

LEVELS = {
    "beginner": "مبتدئ",
    "advanced": "متقدم",
    "research": "بحثي",
}
