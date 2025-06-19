#  Research Paper Assistant

This Colab/CLI-based tool uses a multimodal RAG pipeline to summarize and extract insights from academic PDFs — including text, tables, and images. Powered by LangChain, Unstructured, Gemini 1.5 Flash, and DocArray, it enables intelligent understanding of research papers in one go.

---

### ✅ Repository Includes
📄 A complete Colab/CLI pipeline  
📂 Any academic paper (PDF) with tables, figures, images  
🧰 Utility scripts for loading, summarizing, and embedding  
📦 All required libraries via requirements.txt  
🔑 Uses Google’s Gemini Flash (via LangChain) for generation  

---

## ✨ Features

📝 Accepts research papers in PDF format  
📊 Extracts text, tables, and images  
🧠 Summarises each component using Gemini 1.5 Flash  
🔍 Retrieves content chunks via DocArrayInMemorySearch  
💬 Answers user queries with context-aware RAG  
📸 Supports image captioning from figures/diagrams  
⚡ Efficient multimodal handling using Unstructured  

---

##  Tech Stack

| Component         | Tool / Library                          |
|-------------------|------------------------------------------|
| LLM               | Gemini 1.5 Flash via LangChain           |
| PDF Processing    | Unstructured + Poppler + pdf2image       |
| OCR               | Tesseract (used within Unstructured)     |
| Embeddings        | GoogleGenerativeAIEmbeddings             |
| Vector Search     | DocArrayInMemorySearch                   |
| Text Splitting    | chunking_strategy="by_title" (Unstructured) |
| Image Captioning  | Gemini multimodal prompt via base64      |
| Summarization     | LangChain PromptTemplate + Gemini        |

---

### Example Use Case
"What is positional encoding?"

The assistant extracts relevant content from both text and images (e.g., equations, graphs), providing a concise, accurate summary using Gemini Flash.

---

### ⚙️ How to Use (Colab)
Open the notebook in Google Colab

Upload the research paper PDF

Add your Gemini API key

Run all cells in order.

---

## Setup Instructions(Local)
```bash

# Clone the repository
git clone https://github.com/Tipsy-12/Intelligent-Paper-Assistant.git
cd Intelligent-Paper-Assistant

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

GOOGLE_API_KEY=your_gemini_api_key_here
LANGCHAIN_TRACING_V2=false
```

### ⚙️ System Dependencies
You’ll need the following tools installed on your system:

📦 Linux / Ubuntu / Codespaces:
```bash
sudo apt update && sudo apt install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1
```
poppler-utils: for pdf2image to convert PDFs to images.

tesseract-ocr: for OCR capabilities in unstructured.

libgl1: required by opencv-python.

---



 ### 📁 Project Structure
```bash

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






