import os
from openai import OpenAI

from fastapi import FastAPI
from pydantic import BaseModel


class Request(BaseModel):
    url: str
    questions: list[str]


app = FastAPI()

@app.post("/answer")
async def answer_questions(request: Request):
    return {}

# openai_api_key = os.environ['OPENAI_API_KEY']
# openai_api_base = "https://api.openai.com/v1/"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# chat_response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
# print("Chat response:", chat_response)

