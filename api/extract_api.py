import requests
from bs4 import BeautifulSoup
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

class ExtractRequest(BaseModel):
    url: str = ""

@app.post("/api/extract")
def extract_url_content(item: ExtractRequest):
    url = item.url
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    media_source = 'No media source currently available'
    underline = '______________________'
    article_content = soup.find(id='article-content')
    paragraphs_soup = article_content.find("div", class_="wsw")

    sentences = []
    for paragraph in paragraphs_soup.find_all('p'):
        text = paragraph.text
        if text.startswith(underline):
            break
        elif text != media_source:
            sentences.extend(sent_tokenize(text))

    return_sents = []
    for idx, sent in enumerate(sentences):
        tokens = tokenize_processor(sent)
        return_sents.append({'id': idx, 'tokens': tokens})

    return JSONResponse({'sentences': return_sents})
