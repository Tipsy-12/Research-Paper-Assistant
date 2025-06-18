import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("AIzaSyDcyPkDBHE2HaNe9Hn_HZgu-RhBywWLSl0")
LANGCHAIN_API_KEY = os.getenv("AIzaSyDcyPkDBHE2HaNe9Hn_HZgu-RhBywWLSl0", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
PDF_PATH = "data/your_paper.pdf"