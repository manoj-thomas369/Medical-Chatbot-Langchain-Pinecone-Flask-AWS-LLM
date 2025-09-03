from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.prompt import *
import os


app=Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
embeddings = download_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(LLM, BaseModel):
    model: str = Field(default="gemini-2.5-flash")  # Pydantic field

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Ensure prompt is a string
        if isinstance(prompt, dict) and "input" in prompt:
            prompt = prompt["input"]
        if not prompt:
            prompt = " "
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "gemini"

chatModel = GeminiLLM()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)