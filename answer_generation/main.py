import json
import os
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

RETRIEVER_URL = os.getenv("RETRIEVER_URL")

app = FastAPI()
llm = ChatOpenAI(model='gpt-4o-mini')


prompt = PromptTemplate.from_template(
    """Ты эксперт по строительной документации. Я задам тебе вопрос, а также дам релевантную информацию из документации. """
    """Ты должен ответить на поставленный вопрос на русском языке, используя только информацию из контекста, НИЧЕГО не добавляй от себя. """
    """Вопрос: {question}\nКонтекст: {context}\n"""
)


def answer_question(query: str, docs: list[str]):
    chain = prompt | llm
    context = '\n'.join(docs)
    result = chain.invoke({'question': query, 'context': context})
    return result


@app.get("/ask")
async def ask_question(query: str):
    async with httpx.AsyncClient() as client:
        try:
            query_params = {"query": query, "num": 5}
            response = await client.get(urljoin(RETRIEVER_URL, "search"), params=query_params)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.content.decode())
            docs = json.loads(response.content)['results']['documents']
            return {"response": answer_question(query, docs[0])}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
