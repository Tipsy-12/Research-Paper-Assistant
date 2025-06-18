import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Create model with explicit API key
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Prompt + chain
prompt_template = ChatPromptTemplate.from_template("""
You are an assistant tasked with summarizing tables and text.
Table or text chunk: {element}
""")

summarize_chain = {"element": lambda x: x} | prompt_template | model | StrOutputParser()

# Summary functions
def summarize_texts(text_chunks):
    return summarize_chain.batch(text_chunks, {"max_concurrency": 3})

def summarize_tables(table_chunks):
    html_tables = [t.metadata.text_as_html for t in table_chunks]
    return summarize_chain.batch(html_tables, {"max_concurrency": 3})
