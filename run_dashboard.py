import sys

from streamlit.web import cli as streamlit_cli


sys.argv = [
    "streamlit",
    "run",
    "dashboard.py",
    "--server.port",
    "8501",
    "--server.headless",
    "true",
    "--server.fileWatcherType",
    "none",
]
streamlit_cli.main()
