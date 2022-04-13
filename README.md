# TopSense

## Step 1
Map Roget's thesaurus to Cambridge dictionary   

##### Input
- ./data/BRT_data.json  
#####  Output
- ./data/jsonl_file/{OUTPUT_FILENAME}.jsonl  
##### Program file
- ./data_preprocess/refactor_cross_pos_related.py   
```
python refactor_cross_pos_related.py -f normal
```  

---
## Step 2
Transform Dictionary examples into two kinds of training data 
 (reserve target word and don't reserve target word)   

##### Input
- ./data/jsonl_file/{OUTPUT_FILENAME}.jsonl (Output file from Step 1)
- ./data/cambridge.sense.000.jsonl (Cambridge dictionary data) 
##### Output
- ./data/training_data/{OUTPUT_FILENAME}.tsv   
(origin sentence, topic sentence, masked sentence)
##### Program file
##### Preprocess Cambridge data with mapped category
- ./data_preprocess/process_mapped_data.py
##### Transform into MASK sentence
- ./data_preprocess/process_MASK_data.py

---
## Step 3
Use Simple English Wikipedia data to augment training data 
##### Input
- ./data/wiki/
##### Output 
- ./data/training_data/{OUTPUT_FILENAME}.tsv
(origin sentence, topic sentence, masked sentence)
##### Program file
1. Preprocess Simple English Wikipedia data
2. Transform into MASK sentence
-  ./data_preprocess/process_wiki_sent_MASK.py

---
## Step 4
Fine-tune BertForMaskedLM to predict topics  
### 1. First add topic tokens into tokenizer (caution: cased/uncased)
##### Input
##### Output
- ./tokenizer_{casedTrue/casedFalse}
##### Program file
- ./data_preprocess/gen_tokenizer.py
```
python gen_tokenizer.py -c 0
python gen_tokenizer.py -c 1
```


### 2. Train model
##### Input
- preprocessed training data from **data/training_data**
##### Output 
- a fine-tuned masked language model
##### Program file
```
python train_MLM.py
```

### 3. Evaluation
- 100 sents
- 300 sents
### 3-1 Sample evaluation sentences from Voice of America (VOA) and Macmillian English Dictionary (MED)
##### Input
##### Output
##### Program file

### 3-2 Pre-calculate topic embeddings
##### Input
##### Output
##### Progrma file

---
### TODO List
#### 1. Add Wikipedia sentences of words not in Cambridge dictionary into training data
#### 2. Add Wikipedia sentences of monosemous word into trainig data 
#### 3. Other POS tags





