# TopSense

## Step 1
Map Roget's thesaurus to Cambridge dictionary   
### 1-1 Use partial BRT data to check the result
WORD = mole/bass/taste/issue/interest/bow/cone/slug/sentence/bank/star/duty
#### Command
Enter this command line below in data_preprocess directory.
```
python map_thesaurus2cambridge.py -f test -w [WORD] -c top3 -r 0 -g 1 
```
#### Input
```
./data/mapping_test_data/BRT_data_{WORD}_test.json
```
####  Output
```
./data/jsonl_file/test_jsonl/True_0.0_top3_{WORD}.jsonl
```

### 1-2 Map all BRT data to Cambridge dictionary
#### Command
```
python map_thesaurus2cambridge.py -f normal -c top3 -r 0 -g 1
```  
#### Input
```
./data/BRT_data.json  
```
####  Output
```
./data/jsonl_file/True_0.0_top3_all.jsonl  
```
---
## Step 2
Transform Dictionary examples into two kinds of training data  
 (reserve target word and don't reserve target word)    
### 2-1 Preprocess Cambridge data with mapped category
#### Command
Enter this command line below in data_preprocess directory.
```
python process_mapped_data.py -t 0
```
#### Input
```
./data/jsonl_file/True_0.0_top3_all.jsonl 
./data/cambridge.sense.000.jsonl
```
#### Output
```
./data/0.word_id.topics.examples.json
```
### 2-2 Transform into MASK sentence
#### Command
Enter this command line below in data_preprocess directory.
```
python process_MASK_data.py -r 0
python process MASK_data.py -r 1
```
#### Input
```
./data/0.word_id.topics.examples.json
```
#### Output
data in file: origin sentence, topic sentence, masked sentence
```
./data/training_data/False_cambridge.tsv
./data/training_data/True_cambridge.tsv  
```


- ./data_preprocess/process_MASK_data.py

---
## Step 3
Use Simple English Wikipedia data to augment training data.
### 3-1 Preprocess Simple English Wikipedia data 
Fetcg the first sentences in linked pages, collect sentences that be linked to the page
#### Command
#### Input
#### Output
### 3-2 Transform into MASK sentences
#### Command
#### Input
- ./data/wiki/
#### Output 
- ./data/training_data/{OUTPUT_FILENAME}.tsv
(origin sentence, topic sentence, masked sentence)
#### Program file
1. Preprocess Simple English Wikipedia data
2. Transform into MASK sentence
-  ./data_preprocess/process_wiki_sent_MASK.py
### 3-2 Concatenate Cambridge training data with Wikipedia training data
#### Command
#### Input
#### Output


---
## Step 4
Fine-tune BertForMaskedLM to predict topics  
### 4-1 First add topic tokens into tokenizer (caution: cased/uncased)
#### Command
#### Input
#### Output
- ./tokenizer_{casedTrue/casedFalse}
#### Program file
- ./data_preprocess/gen_tokenizer.py
```
python gen_tokenizer.py -c 0
python gen_tokenizer.py -c 1
```


### 2. Train model
#### Input
- preprocessed training data from **data/training_data**
#### Output 
- a fine-tuned masked language model
#### Program file
```
python train_MLM.py
```

### 3. Evaluation
- 100 sents
- 300 sents
### 3-1 Sample evaluation sentences from Voice of America (VOA) and Macmillian English Dictionary (MED)
#### Input
#### Output
#### Program file

### 3-2 Pre-calculate topic embeddings
#### Input
#### Output
#### Progrma file

---
### TODO List
#### 1. Add Wikipedia sentences of words not in Cambridge dictionary into training data
#### 2. Add Wikipedia sentences of monosemous words into trainig data 
#### 3. Other POS tags





