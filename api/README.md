## TopSense API
=============================
TopSense is a WSD model that disambiguate word sense based on topics.   
TopSense API can input a sentence and get the wsd results of nouns and verbs  

Structure
-------------------
```
/
 |-- api/
 |-- TopSense 
 |    |-- zip_file     (<------ put downloaded zip file here and unzip it)
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
python -m spacy download en_core_web_md
```

### Download data  
 + **manually**  
   Download the zip file and unzip this file under TopSense directory.  
   The needed data for WSD and WSD models, tokenizer are included in this file.
   - [zip file]()

### Run the API
run this command under the api folder
```
CUDA_VISIBLE_DEVICES="" uvicorn --host 0.0.0.0 --port 5555 topsense:app
```

