import requests
from bs4 import BeautifulSoup
from readability import Document
from nltk.tokenize import sent_tokenize
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
sys.path.append('..')
from TopSense.util import tokenize_processor

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

def ValidURL(url):
    return url and url.strip().startswith('http')

class ExtractRequest(BaseModel):
    query: str = ""

@app.post("/api/extract")
def extract_url_content(item: ExtractRequest):
    input = item.query
    title = ''
    return_sents = []
    
    if ValidURL(input):
        url = input
        response = requests.get(url)
        doc = Document(response.text)
        title = doc.title()
        soup = BeautifulSoup(doc.summary(), 'lxml')

        sentences = []
        for paragraph in soup.find_all('p'):
            text = paragraph.text.strip()
            sentences.extend(sent_tokenize(text))
        
        for idx, sent in enumerate(sentences):
            tokens = tokenize_processor(sent)
            return_sents.append({'id': idx, 'tokens': tokens})
        
    else:
        sentences = sent_tokenize(input)
        for idx, sent in enumerate(sentences):
            tokens = tokenize_processor(sent)
            return_sents.append({'id': idx, 'tokens': tokens})

    return JSONResponse({'title': title, 'sentences': return_sents})
