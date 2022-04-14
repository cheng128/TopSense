# TopSense
A Word Sence Disambiguation System.
## Step 1
Map Roget's thesaurus (BRT) to Cambridge dictionary   
### 1-1 Use partial BRT data to check the result
WORD = mole/bass/taste/issue/interest/bow/cone/slug/sentence/bank/star/duty
#### Command
Run this command in data_preprocess directory.
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
Run this command in data_preprocess directory.
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
Run this command in data_preprocess directory.
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
---
## Step 3
Use Simple English Wikipedia data to augment training data.
Run these commands in data_preprocess directory.
### 3-1 Preprocess Simple English Wikipedia data 
Fetch the first sentences in linked pages, collect sentences that are linked to the page
#### Command
```
python build_wikipedia_data_map.py -v simple
```
#### Input
```
./data/words2defs.json
./data/0.word_id.topics.examples.json
./data/wiki/simple_en_wiki_link.txt
```
#### Output
```
./data/wiki/simple_wiki_href2def.json
./data/wiki/simple_wiki_href_word2sents.json
```
### 3-2 Sample sentences in Wikipedia
NUM = 10/20
#### Command
```
python sample_wiki_sents.py -v simple -n [NUM]
```
#### Input
```
./data/wiki/all_simple_wiki_id2sents.json
```
#### Output
```
./data/wiki/[NUM]_simple_wiki_id2sents.json
```
### 3-3 Transform into MASK sentences
#### Command
```
python process_wiki_sent_MASK.py -r 0 -n [NUM] -v simple
python process_wiki_sent_MASK.py -r 1 -n [NUM] -v simple
```
#### Input
```
./data/wiki/[NUM]_simple_wiki_id2sents.json
```
#### Output 
```
```
### 3-4 Concatenate Cambridge training data with Wikipedia training data
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


### 4-2 Train model
#### Input
- preprocessed training data from **data/training_data**
#### Output 
- a fine-tuned masked language model
#### Program file
```
python train_MLM.py
```

### 4-3 Evaluation
- 100 sents
- 300 sents

### 4-3-1 Sample evaluation sentences from Voice of America (VOA) and Macmillian English Dictionary (MED)
#### Input
#### Output
#### Program file

### 4-3-2 Pre-calculate topic embeddings
#### Input
#### Output
#### Progrma file

### 4-4 Best model now 
learning rate: 1e-05  
batch size: 64  
- First train with the file below for about 15 epochs:
```
training_data/no_reserve/0.6_remap_20_False_concat.tsv
```
#### Command
```
python train_MLM_with_validation.py -e 15 -g concat -f training_data/no_reserve/0.6_remap_20_False_concat.tsv -n 20 -r 0 -lr 1e-05 -m 0.6_val_sec
```
#### Output
```
concat/0.6_val_sec_20_False_15epochs_1e-05
```


- Further traing the model with the file below for about 4 epochs
training_data/reserve/0.6_remap_20_True_concat.tsv

---
### TODO List
#### 1. Add Wikipedia sentences of words not in Cambridge dictionary into training data
#### 2. Add Wikipedia sentences of monosemous words into trainig data 
#### 3. Other POS tags





