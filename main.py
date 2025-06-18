from config import PDF_PATH
from utils import pdf_loader, summarizer, image_utils, vector_store, rag_chain

# Load and chunk the PDF
texts, tables, images = pdf_loader.process_pdf(PDF_PATH)

# Generate summaries
text_summaries = summarizer.summarize_texts(texts)
table_summaries = summarizer.summarize_tables(tables)
image_summaries = image_utils.summarize_images(images)

# Build vector store + retriever
retriever = vector_store.build_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries)

# Run QA chain
response = rag_chain.run_query("What is positional encoding?", retriever)

# Output
print("Answer:", response["response"])