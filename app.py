import os
import tempfile
import uuid
import base64
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf

 # Replace this securely (e.g., via .env)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC5-KL5y09UdsWpLVWnTFKfh49kMfU_J2s"  # Use dotenv for security 

st.set_page_config(page_title="🧠 Intelligent Paper Assistant", layout="wide")
st.title("📄 Intelligent Paper Assistant")

uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

def extract_pdf_elements(pdf_path):
    chunks = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    texts, tables = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
    return texts, tables, images

def summarize_chunks(model, chunks):
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary.
    Table or text chunk: {element}
    """)
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return chain.batch(chunks, {"max_concurrency": 3})

def summarize_images(model, images_b64):
    summaries = []
    for img in images_b64:
        message = [HumanMessage(content=[
            {"type": "text", "text": "Describe the image in detail. The image is from a paper explaining transformer architecture."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        ])]
        chain = model | StrOutputParser()
        summaries.append(chain.invoke(message))
    return summaries

def docset(docs, summaries, key="doc_id"):
    ids = [str(uuid.uuid4()) for _ in docs]
    docs_meta = [Document(page_content=s, metadata={key: ids[i]}) for i, s in enumerate(summaries)]
    return docs_meta, list(zip(ids, docs))

def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            base64.b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    ctx = kwargs["context"]
    question = kwargs["question"]
    context_text = "".join([x.text for x in ctx["texts"]])
    prompt = [{"type": "text", "text": f"""
    Answer the question using this context (text + table + image summaries):
    Context: {context_text}
    Question: {question}
    """}]
    for img in ctx["images"]:
        prompt.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    return [HumanMessage(content=prompt)]

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    with st.spinner("⏳ Extracting content..."):
        texts, tables, images = extract_pdf_elements(temp_pdf_path)

    st.write("### 📄 Raw Extracted Content")
    st.write("**Text Chunks:**", len(texts))
    st.write("**Table Chunks:**", len(tables))
    st.write("**Image Chunks:**", len(images))

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    with st.spinner("🧠 Summarizing text..."):
        text_summaries = summarize_chunks(model, texts)
    with st.spinner("📊 Summarizing tables..."):
        table_html = [table.metadata.text_as_html for table in tables]
        table_summaries = summarize_chunks(model, table_html)
    with st.spinner("🖼️ Summarizing images..."):
        image_summaries = summarize_images(model, images)

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

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    summary_text_docs, text_pairs = docset(texts, text_summaries)
    summary_table_docs, table_pairs = docset(tables, table_summaries)
    summary_image_docs, image_pairs = docset(images, image_summaries)

    vectorstore = DocArrayInMemorySearch.from_documents(summary_text_docs, embedding=embedding_function)
    vectorstore.add_documents(summary_table_docs)
    vectorstore.add_documents(summary_image_docs)

    store = InMemoryStore()
    store.mset(text_pairs + table_pairs + image_pairs)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    chain_with_sources = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough().assign(
            response=(RunnableLambda(build_prompt) | model | StrOutputParser())
        )
    )

    user_query = st.text_input("🔍 Ask a question about the paper:")
    if user_query:
        with st.spinner("🤖 Thinking..."):
            result = chain_with_sources.invoke(user_query)
            st.subheader("🧠 Answer")
            st.write(result["response"])

            with st.expander("🧾 Context used"):
                for ctx in result["context"]["texts"][:3]:
                    st.markdown(ctx.text[:500] + "...")
                if result["context"]["images"]:
                    st.subheader("🖼️ Images Used in Context")
                    for img_b64 in result["context"]["images"]:
                        st.image(f"data:image/jpeg;base64,{img_b64}", use_column_width=True)

    os.remove(temp_pdf_path)


