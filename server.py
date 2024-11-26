import io
import os
from openai import OpenAI
import ulid

from thefuzz import fuzz

from fastapi import FastAPI
from pydantic import BaseModel

import requests

from pypdf import PdfReader

from inmemorydb import InMemoryVectorDB

memorydb = InMemoryVectorDB()

openai_api_key = os.environ['OPENAI_API_KEY']
openai_api_base = "https://api.openai.com/v1/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


class Request(BaseModel):
    url: str
    questions: list[str]

# Tokenization and Vocabulary Building
def tokenize_documents(docs):
    vocabulary = set()
    tokenized_docs = [doc.lower().split() for doc in docs]
    for doc in tokenized_docs:
        vocabulary.update(doc)
    vocabulary = sorted(list(vocabulary))
    return tokenized_docs, vocabulary

def vectorize_documents(docs, vocab):
    vectors = []
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    for doc in docs:
        vector = [0] * len(vocab)
        for word in doc:
            if word in vocab_index:
                vector[vocab_index[word]] += 1
        vectors.append(vector)
    return vectors


app = FastAPI()

@app.post("/answer")
async def answer_questions(request: Request):
    resp = requests.get(request.url)
    # print(resp.text)
    on_fly_mem_obj = io.BytesIO(resp.content)
    pdf_file = PdfReader(on_fly_mem_obj)
    
    number_of_pages = len(pdf_file.pages)
    page = pdf_file.pages[0]
    text = page.extract_text()

    requestid = ulid.new()
    collection = memorydb.get_or_create_collection(requestid)

    documents = []
    documentids = []
    page_count = 0
    for page in pdf_file.pages:
        text = page.extract_text() 
        # print(text)
        documents.append(text)
        documentids.append("page-" + str(page_count))
        page_count += 1

    tokenized_documents, vocabulary = tokenize_documents(documents)
    bow_embeddings = vectorize_documents(tokenized_documents, vocabulary)

    collection.add(
        embeddings=bow_embeddings,
        documents=documents,
        ids=documentids,
    )

    for question in request.questions:
        print("question:", question)
        # search_query = "EUR to INR conversion rate on 17-03-2024"
        tokenized_query, _ = tokenize_documents([question])
        query_embeddings = vectorize_documents(tokenized_query, vocabulary)

        results = collection.query(
            query_embeddings=query_embeddings,
        )
        print(results)
    
    return {
        "pages": number_of_pages,
        "text": text,
    }


# chat_response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
# print("Chat response:", chat_response)

