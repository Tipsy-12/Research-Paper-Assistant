import uuid
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
id_key = "doc_id"

def build_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries):
    vectorstore = DocArrayInMemorySearch()
    store = InMemoryStore()

    def add(docs, summaries):
        ids = [str(uuid.uuid4()) for _ in docs]
        summary_docs = [Document(page_content=summary, metadata={id_key: ids[i]}) for i, summary in enumerate(summaries)]
        vectorstore.add_documents(summary_docs)
        store.mset(list(zip(ids, docs)))

    add(texts, text_summaries)
    add(tables, table_summaries)
    add(images, image_summaries)

    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    return retriever