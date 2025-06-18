import streamlit as st
from utils import pdf_loader, summarizer, image_utils, vector_store
from config import PDF_PATH
import tempfile
import os

st.set_page_config(page_title="🧠 Intelligent Paper Assistant", layout="wide")
st.title("📄 Intelligent Paper Assistant")

uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # Extract elements from PDF
    with st.spinner("⏳ Extracting content..."):
        texts, tables, images = pdf_loader.partition_pdf(temp_pdf_path)

    st.write("### 📄 Raw Extracted Content")
    st.write("**Text Chunks:**", len(texts))
    st.write("**Table Chunks:**", len(tables))
    st.write("**Image Chunks:**", len(images))

    with st.spinner("🧠 Summarizing text..."):
        text_summaries = summarizer.summarize_texts(texts)
    with st.spinner("📊 Summarizing tables..."):
        table_summaries = summarizer.summarize_tables(tables)
    with st.spinner("🖼️ Summarizing images..."):
        image_summaries = image_utils.summarize_images(images)

    with st.spinner("🔍 Building retriever..."):
        retriever = vector_store.build_retriever(
            texts, tables, images,
            text_summaries, table_summaries, image_summaries
        )

    st.success("✅ Summaries generated!")
    st.subheader("📌 Text Summary Samples")
    for summary in text_summaries[:3]:
        st.markdown(f"- {summary}")

    st.subheader("📌 Table Summary Samples")
    for summary in table_summaries[:3]:
        st.markdown(f"- {summary}")

    st.subheader("📌 Image Summary Samples")
    for summary in image_summaries[:3]:
        st.markdown(f"- {summary}")

    os.remove(temp_pdf_path)
