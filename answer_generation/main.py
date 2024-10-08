import json
import logging
import os
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RefineDocumentsChain, LLMChain
from langchain.docstore.document import Document

from prompts import prompt_template, refine_prompt_template, rewrite_query_prompt_template

logging.basicConfig(level=logging.INFO)

load_dotenv()

RETRIEVER_URL = os.getenv("RETRIEVER_URL")

app = FastAPI()
llm = ChatOpenAI(model='gpt-4o-mini')


def answer_question(query: str, docs: list[str], approach: str = 'concat'):
    if approach == 'concat':
        chain = prompt_template | llm
        context = '\n'.join(docs)
        result = chain.invoke({'question': query, 'context': context})
        return result
    elif approach == 'refine':
        docs = [Document(page_content=doc) for doc in docs]
        chain = RefineDocumentsChain(
            initial_llm_chain=LLMChain(prompt=prompt_template, llm=llm),
            refine_llm_chain=LLMChain(prompt=refine_prompt_template, llm=llm),
            document_variable_name="context",
            initial_response_name="existing_answer",
        )
        return chain.run(input_documents=docs, question=query)
    else:
        raise ValueError(f"In function `answer_question` param `approach` can't be {approach}")


@app.get("/ask")
async def ask_question(query: str, num: int = 5, approach: str = 'concat', reranker: bool = True):
    chain = rewrite_query_prompt_template | llm
    multi_query = chain.invoke({'query': query}).content
    async with httpx.AsyncClient(timeout=1200) as client:
        try:
            query_params = {"query": query, "num": num, "reranker": reranker}
            response = await client.get(urljoin(RETRIEVER_URL, "search"), params=query_params)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.content.decode())
            docs = json.loads(response.content)['docs']
            return {"response": answer_question(query, docs, approach=approach)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
