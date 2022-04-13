# TopSense

## Step 1
Map Roget's thesaurus to Cambridge dictionary   

### Input
- data/BRT_data.json  
### Output
- data/jsonl_file/{OUTPUT_FILENAME}.jsonl  
### Program file
- data_preprocess/refactor_cross_pos_related.py   
```
python refactor_cross_pos_related.py -f normal
```  

---
## Step 2
Transform Dictionary examples into two kinds of training data (reserve target word and don't reserve target word)   

### Input
- data/jsonl_file/{OUTPUT_FILENAME}.jsonl (Output from Step 1)
- data/cambridge.sense.000.jsonl (Cambridge dictionary data) 
### Output
- data/training_data/{OUTPUT_FILENAME}.tsv (origin sentence, topic sentence, masked sentence)
### Program file
#### Preprocess Cambridge data with mapped category
- data_preprocess/process_mapped_data.py
#### Transform into MASK sentence
- data_preprocess/process_MASK_data.py

---
## Step 3
Use Simple English Wikipedia data to augment training data 
### Input
- Simple English Wikipedia data with links
### Output 

### Program file
#### Preprocess Simple English Wikipedia data
#### Transform into MASK sentence
-  data_preprocess/process_wiki_sent_MASK.py

---
### Step 4
Fine-tune BertForMaskedLM to predict topics  
- first add topic tokens into tokenizer (caution: cased/uncased)
### Input
### Output 
### Program file
