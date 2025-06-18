from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def build_prompt_for_image(base64_img):
    return [
        HumanMessage(content=[
            {"type": "text", "text": "Describe the image in detail. It is part of a research paper explaining transformer architecture."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ])
    ]

def summarize_images(images):
    chain = model | StrOutputParser()
    return [chain.invoke(build_prompt_for_image(img)) for img in images]