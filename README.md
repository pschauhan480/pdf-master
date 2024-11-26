# pdf-master

Reads a pdf and answers questions based on the content of it.

# Instructions to run

1. clone the project and `cd` into the directory
2. setup a virtual environment and install all packages from the given requirements.txt: `pip install -r requirements.txt`
3. run the server: `uvicorn server:app --host 0.0.0.0 --port 4000`
4. make request over the REST api using following structure:

## Example

- POST http://localhost:4000/answer

### Request

```json
{
  "url": "<public pdf url>",
  "questions": [
    "What is the name of the company?",
    ...
  ]
}
```

### Response

```json
[
  {
    "question": "What is the name of the company?",
    "answer": "Kisai"
  },
  ...
]
```
