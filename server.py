# Global Imports
import io
import os
import json
from openai import OpenAI
import ulid

from thefuzz import fuzz

from fastapi import FastAPI
from pydantic import BaseModel

import requests

from pypdf import PdfReader

from inmemorydb import InMemoryVectorDB

# initialize vector db for embeddings
memorydb = InMemoryVectorDB()

# llm client, can be initialized to point to self hosted instances too
openai_api_key = os.environ['OPENAI_API_KEY']
openai_api_base = "https://api.openai.com/v1/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# can be made configuration
minimum_confidence_score = 0.75

# request model for validations
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
    requestid = ulid.new()
    try:
        resp = requests.get(request.url)
        # print(resp.text)
        
        # read pdf from url and convert to text for embeddings
        on_fly_mem_obj = io.BytesIO(resp.content)
        pdf_file = PdfReader(on_fly_mem_obj)
        
        page = pdf_file.pages[0]
        text = page.extract_text()

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

        final_responses = []

        for question in request.questions:
            print("question:", question)
            # embed question and match for similarity with data of pdf
            question_response = {
                "question": question,
            }
            tokenized_query, _ = tokenize_documents([question])
            query_embeddings = vectorize_documents(tokenized_query, vocabulary)

            results = collection.query(
                query_embeddings=query_embeddings,
            )
            # print(results)

            context = ""
            for document in results['documents']:
                context += document

            # append context to the question being asked by the user
            chat_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Based on this given text: \"{context}\" answer the question: \"{question}\" and give confidence score also of how much you think the answer is correct in the format of json with two keys as 'answer' and 'score'. Don't attach any markdown specific tags. just return raw json as text".format(context=context, question=question)},
                ]
            )
            # print( chat_response)

            # check for confidence score and form final response
            if len(chat_response.choices) > 0:
                print("Chat response:", chat_response.choices[0].message.content)
                choice = json.loads(str(chat_response.choices[0].message.content))
                if choice["score"] < minimum_confidence_score:
                    question_response["answer"] = "Data Not Available"
                else:
                    question_response["answer"] = choice["answer"]
            else:
                question_response["answer"] = "Data Not Available"
            final_responses.append(question_response)
        
        return {
            "id": str(requestid),
            "response": final_responses
        }
    except Exception as e:
        print("error occurred:", e)
        return {
            "id": str(requestid),
            "response": "something went wrong"
        }
