#  Research Paper Assistant

An AI-powered assistant that reads, analyzes, and summarizes research papers including **PDFs, tables, and images** using **LangChain**, **Gemini 1.5 Flash**, **Unstructured**, and **DocArray**.

---

## ✨ Features

- Extracts and summarizes **text**, **tables**, and **images** from academic papers.
- Uses **Gemini 1.5 Flash** (via LangChain) for high-quality summarization.
- Retrieves information using **DocArrayInMemorySearch**.
- Processes documents using **Unstructured** for PDF + image handling.

---

## 🚀 Setup Instructions

### 1. ✅ Clone the Repo

```bash
git clone https://github.com/Tipsy-12/Intelligent-Paper-Assistant.git
cd Intelligent-Paper-Assistant

### 2. ✅ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

### 3. ✅ Install Python dependencies
pip install -r requirements.txt

### 4. Set up environment variables
GOOGLE_API_KEY=your_gemini_api_key_here
LANGCHAIN_TRACING_V2=false

###⚙️ System Dependencies (IMPORTANT)
You’ll need the following tools installed on your system:

📦 Linux / Ubuntu / Codespaces:
bash
Copy
Edit
sudo apt update && sudo apt install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1
poppler-utils: for pdf2image to convert PDFs to images.

tesseract-ocr: for OCR capabilities in unstructured.

libgl1: required by opencv-python.

🧪 Run the App
bash
Copy
Edit
python main.py
Make sure your .env is set, and PDF_PATH in config.py points to your target PDF.

🧰 Tech Stack
Component	Library/Tool
LLM	Gemini 1.5 Flash via langchain-google-genai
OCR + Layout	Unstructured + Tesseract + Poppler
Vector Search	DocArrayInMemorySearch + LangChain
Embeddings	GoogleGenerativeAIEmbeddings
PDF Processing	pdf2image, pypdf, unstructured
Summarization	LangChain PromptTemplates + Gemini

📁 Project Structure
bash
Copy
Edit
├── main.py
├── config.py
├── requirements.txt
├── .env                 # Not committed (contains API keys)
├── utils/
│   ├── pdf_loader.py
│   ├── summarizer.py
│   ├── image_utils.py
│   ├── vector_store.py
├── .gitignore






