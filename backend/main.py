import uvicorn
from fastapi import FastAPI
from model import generate_response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware



class PromptRequest(BaseModel):
    prompt : str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # Allow your Express server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello():
    return "hello world"

@app.post('/submit')
def generate_text(request:PromptRequest):
    return generate_response(request.prompt)
    
    
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)

