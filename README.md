# TopSense

## Step 1
Map Roget's thesaurus to Cambridge dictionary   

### Input
- dir: data/BRT_data.json  
### Output
- dir: data/jsonl_file/{OUTPUT_FILENAME}.jsonl  
### Program file
- data_preprocess/refactor_cross_pos_related.py   
```
python refactor_cross_pos_related.py -f normal
```  

## Step 2
Transform Dictionary examples into two kinds of training data (reserve target word and don't reserve target word)   

### Program file
#### Preprocess Cambridge data with mapped category
- data_preprocess/process_mapped_data.py
#### Transform into MASK sentence
- data_preprocess/process_MASK_data.py

## Step 3
Use Simple English Wikipedia data to augment training data  
### Program file
-  data_preprocess/process_wiki_sent_MASK.py

### Step 4
Fine-tune BertForMaskedLM to predict topics  
- first add topic tokens into tokenizer (caution: cased/uncased)