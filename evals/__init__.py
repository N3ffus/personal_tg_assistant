"""Prompt and integration evaluations, separate from runtime dependencies."""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "NO")
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
