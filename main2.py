import os
import uuid
import base64
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf

# === Config ===
os.environ["GOOGLE_API_KEY"] = "your-api-key"  # Replace or use dotenv
file_path = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
id_key = "doc_id"

# === Load PDF and Chunk ===
chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)

# === Split Types ===
texts, tables = [], []
for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

def extract_images(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = extract_images(chunks)

# === Summarization ===
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
summary_prompt = ChatPromptTemplate.from_template("""
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary.
Table or text chunk: {element}
""")
summarize_chain = {"element": lambda x: x} | summary_prompt | model | StrOutputParser()
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
table_html = [table.metadata.text_as_html for table in tables]
table_summaries = summarize_chain.batch(table_html, {"max_concurrency": 3})

def build_image_prompt(base64_img):
    return [HumanMessage(content=[
        {"type": "text", "text": "Describe the image in detail. The image is from a paper explaining transformer architecture."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
    ])]

image_summaries = []
for img in images:
    prompt = build_image_prompt(img)
    chain = model | StrOutputParser()
    summary = chain.invoke(prompt)
    image_summaries.append(summary)

# === Store and Embed ===
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def docset(docs, summaries):
    ids = [str(uuid.uuid4()) for _ in docs]
    docs_meta = [Document(page_content=s, metadata={id_key: ids[i]}) for i, s in enumerate(summaries)]
    return docs_meta, list(zip(ids, docs))

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
    id_key=id_key,
)

# === RAG Prompt Builder ===
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
    Answer the question using this context (includes text, tables, and image(s)):
    Context: {context_text}
    Question: {question}
    """}]
    for img in ctx["images"]:
        prompt.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    return [HumanMessage(content=prompt)]

# === Chain with Source Context ===
chain_with_sources = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnablePassthrough().assign(
        response=(RunnableLambda(build_prompt) | model | StrOutputParser())
    )
)

if __name__ == "__main__":
    query = "What is positional encoding?"
    result = chain_with_sources.invoke(query)
    print("\nAnswer:\n", result["response"])
    print("\n\nContext:\n")
    for txt in result["context"]["texts"]:
        print(txt.text[:500], "\n---")
