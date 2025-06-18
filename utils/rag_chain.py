from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode

def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = "".join([t.text for t in docs_by_type["texts"]])
    prompt_content = [{"type": "text", "text": f"""
Answer the question based only on the following context, which includes text and image(s).
Context: {context_text}
Question: {user_question}
"""}]
    for image in docs_by_type["images"]:
        prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    return [HumanMessage(content=prompt_content)]

def run_query(question, retriever):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough().assign(
            response=(
                RunnableLambda(build_prompt)
                | ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                | StrOutputParser()
            )
        )
    )
    return chain.invoke(question)