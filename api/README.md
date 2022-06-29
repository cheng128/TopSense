TopSense API
=============================  
TopSense is a WSD model that disambiguates word sense based on topics.   
TopSense API can input a sentence and get the wsd results of nouns and verbs  

Structure
-------------------
```
/ BASE_DIR
 |-- api/
 |-- TopSense 
 |    |-- data    (<------ unzip the downloaded file then put the data folder here)
 |    |-- model    (<------ unzip the downloaded file then put the model folder here)
 |    |-- tokenizer_caseFalse    (<------ unzip the downloaded file then put the tokenizer folder here)
 |    |-- data_class.py
 |    |-- disambiguator_class.py
 |    '-- util.py
 '-- requirements.txt
```

Getting start
-------------------
### Install python package
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Download data  
 + **manually** 
   Download the file, unzip it and move the directories into TopSense directory.  
   The needed data for WSD and WSD models, tokenizer are included in this file.
   - [zip file](https://drive.google.com/file/d/1oPQ6OVyOFac2Jfxlc5DtOlA5Qwlq-h1Y/view?usp=sharing)

### Run the API
Run this command under the api folder.
May take a long time to download some needed data for the model at the first time.
```
CUDA_VISIBLE_DEVICES="" uvicorn --host 0.0.0.0 --port 5555 topsense:app
```

Routing
-------------------
### POST `api/wsd`
required parameters:
- sentence: `String`

returned value:
- sentence: List<[TOKEN_DATA](#TOKEN_DATA)>

#### TOKEN_DATA
```
{
  text: String,
  lemma: String,
  pos: String,
  (if have) wsd: WSD_DATA
}
```

#### WSD_DATA
```
{
  topics: List<TOPIC_DATA>,
  senses: List<SENSE_DATA> 
}
```
#### TOPIC_DATA
```
{
  topic: String,
  class: String,
  score: String
}
```

#### SENSE_DATA
```
{
  en_def: String,
  ch_def: String,
  level: String,
  score: String,
  guideword: String
}
