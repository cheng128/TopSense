# TopSense
A Word Sence Disambiguation System.
## Step 1
Map Roget's thesaurus (BRT) to Cambridge dictionary   
### 1-1 Use part of BRT data to check the result
WORD = mole/bass/taste/issue/interest/bow/cone/slug/sentence/bank/star/duty
#### Command
Run this command under data_preprocess directory.
```
python map_thesaurus2cambridge.py -f test -w [WORD] -c top3 -r 0 -g 1 
```
#### Input
```
./data/mapping_test_data/BRT_data_{WORD}_test.json
```
####  Output
```
./data/jsonl_file/test_jsonl/True_0_top3_{WORD}.jsonl
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
Run this command under data_preprocessdirectory.
The final model used the data that set the -t argument to 0.6
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
./data/0.0.word_id.topics.examples.json
```
### 2-2 Transform into MASK sentence
#### Command
Run this command under data_preprocess directory.
```
python process_MASK_data.py -r 0
python process MASK_data.py -r 1
```
#### Input
```
./data/0.0.word_id.topics.examples.json
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
Run these commands under data_preprocess directory.
### 3-1 Preprocess Simple English Wikipedia data 
Fetch the first sentences in linked pages, collect sentences that are linked to the page
#### Command
```
python build_wikipedia_data_map.py -v simple
```
#### Input
```
./data/0.0.word_id.topics.examples.json
./data/wiki/simple_en_wiki_link.txt
```
#### Output
```
./data/wiki/simple_wiki_href2def.json
./data/wiki/simple_wiki_href_word2sents.json
```
### 3-2 Map Simple English Wikipedia page to Cambridge dictionary
### 3-2-1 Pre-calculate the embeddings of definition and examples
#### Command
```
python cal_sense_examples_emb.py
```
#### Input
```
./data/cambridge.sense.000.jsonl
```
#### Output
```
./data/sentence-t5-xl_sense_examples_embs.pickle
```
#### 3-2-2 Map Wikipedia link page to Cambridge sense
#### Command
``` 
python wiki_link_sent2cam.py -v simple
```
#### Input
```
./data/words2defs.json
./data/sentence-t5-xl_sense_examples_embs.pickle
./data/wiki/simple_wiki_href2def.json
./data/wiki/simple_wiki_href_word2sents.json
./data/cambridge.sense.000.jsonl
```
#### Output
```
./data/wiki/all_simple_wiki_word_id2sents_highest.json
```
### 3-3 Sample sentences in Wikipedia
NUM = 10/20
#### Command
```
python sample_wiki_sents.py -v simple -n [NUM]
```
#### Input
```
./data/wiki/all_simple_wiki_word_id2sents_highest.json
```
#### Output
```
./data/wiki/{NUM}_simple_wiki_word_id2sents_highest.json.json
```
### 3-4 Transform into MASK sentences
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
./data/training_data/no_reserve/{NUM}_simple_False_highest.tsv
./data/training_data/reserve/{NUM}_simple_True_highest.tsv/
```
### 3-5 Concatenate Cambridge training data with Wikipedia training data
#### Command
RESERVE = 0/1 
RESERVE DIRECTORY = no_reserve/reserve
NUM = 10/20/all
```
python concat_training_data.py -v simple -r [RESERVE] -n [NUM]
```
#### Input
```
./data/training_data/{RESERVE}_cambridge.tsv
./data/training_data/{RESERVE DIRECTORY}/{NUM}_simple_{RESERVE}_highest.tsv
```
#### Output
```
./data/training_data/{RESERVE DIRECTORY}/{NUM}_{RESERVE}_concat_highest.tsv
```

---
## Step 4
Fine-tune BertForMaskedLM to predict topics    
### 4-1 First add topic tokens into tokenizer (caution: cased/uncased)
#### Command
```
python gen_tokenizer.py -c 0
```
#### Input
```
./data/BRT_data.json
```
#### Output
```
./tokenizer_casedFalse
```
### 4-2 Train model
#### Input
- preprocessed training data from **data/training_data**
#### Output 
- a fine-tuned masked language model
#### Program file
```
python train_MLM_further.py -e 15 -g remap/concat -f training_data/no_reserve/0.6_remap_20_False_concat.tsv -n 20 -r 0
```

### 4-3 Best model now 
learning rate: 1e-05  
batch size: 64  
#### 4-3-1 First train with the file below for about 15 epochs:
```
training_data/no_reserve/0.6_remap_20_False_concat.tsv
```
#### Command
```
python train_MLM_with_validation.py -e 15 -g concat -f training_data/no_reserve/0.6_remap_20_False_concat.tsv -n 20 -r 0 -lr 1e-05 -m 0.6_val_sec
```
#### Input
```
./data/training_data/no_reserve/0.6_remap_20_False_concat.tsv
```
#### Output
```
./model/concat/0.6_val_sec_20_False_15epochs_1e-05
```
#### 4-3-2 Further traing the model with the file below for about 4 epochs
```
training_data/reserve/0.6_remap_20_True_concat.tsv
```
#### Command
```
python train_MLM_with_validation.py -e 4 -g hybrid -f training_data/reserve/0.6_remap_20_True_concat.tsv -n 20 -r 1 -lr 1e-05 -m wiki_reserve -pre ./model/concat/0.6_val_sec_20_False_15epochs_1e-05
```
#### Input
```
./model/concat/0.6_val_sec_20_False_15epochs_1e-05
```
#### Output
```
./model/hybrid/wiki_reserve_20_True_4epochs_1e-05
```
### 4-4 Evaluation
Run this command under evaluation directory.
Evalute model with MAP top1 and MRR scores.
#### Command
```
python evaluation_formula.py -f 300 -m hybrid/wiki_reserve_20_True_4epochs_1e-05 -w 1 -r 1
```
#### Input
```
./evaluation/data/300_sentences.json
./evaluation/data/300_sentences_ans.json
```
#### Output
```
./evaluation/results/hybrid/wiki_reserve_20_True_4epochs_1e-05_300_reweightTrue_reserveTrue_sentence-t5-xl_formula.tsv
```
---
### TODO List
#### 1. Add Wikipedia sentences of words not in Cambridge dictionary into training data
#### 2. Add Wikipedia sentences of monosemous words into trainig data 
#### 3. Other POS tags





