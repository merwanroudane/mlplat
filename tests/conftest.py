import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
os.environ["ML_STRICT"] = "1"
os.environ["STREAMLIT_CLIENT_SHOW_ERROR_DETAILS"] = "full"
