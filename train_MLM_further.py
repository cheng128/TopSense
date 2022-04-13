import torch
import json
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForMaskedLM, AdamW

class TopicDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def gen_data_loader(masked_sent, topic_sent, tokenizer):
    inputs = tokenizer(masked_sent,
                       return_tensors='pt',
                       max_length=64,
                       truncation=True,
                       padding='max_length')
    
    inputs['labels'] = tokenizer(topic_sent,
                                 return_tensors='pt',
                                 max_length=64,
                                 truncation=True,
                                 padding='max_length')["input_ids"]

    dataset = TopicDataset(inputs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

def load_preprocessed(filename):
    topic_sent = []
    masked_sent = []

    print('read file:', filename)
    with open('./data/' + filename) as f:
        for line in f.readlines():
            pieces = line.split('\t')
            topic_sent.append(pieces[1])
            masked_sent.append(pieces[2])
    return topic_sent, masked_sent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="#epochs", type=int)
    parser.add_argument('-f', type=str) # preprocess file name
    parser.add_argument('--save', action="store_true")
    parser.add_argument('-n', type=str, default='all')
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-m', type=str, default='')
    parser.add_argument('-g', type=str, default='hybrid')
    args = parser.parse_args()
    
    epochs = args.e
    filename = args.f
    reserve = bool(args.r)
    lr_rate = args.lr
    mark = args.m
    print('lr_rate:', lr_rate)

    
    if args.save:
        dir_path = f'./model/remap/{args.g}/{mark}_{args.n}_{reserve}_{epochs}epochs_{lr_rate}'
        try:
            os.mkdir(dir_path)
        except:
            dir_path += '_' + str(1)
            os.mkdir(dir_path)
        print('model will be saved in: ', dir_path)
        
    tokenizer_name = 'remap_tokenizer'
        
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print("use tokenizer:", tokenizer_name)
    
    base_pretrained_model =  './model/remap/concat/0.6_val_sec_20_False_15epochs_1e-05'
    # base_pretrained_model =  './model/remap/concat/0.6_val_sec_20_False_20epochs_5e-06' 
    # base_pretrained_model = './model/remap/concat/0.6_val_all_False_30epochs_1e-05'
    print('pretrained:')
    model = BertForMaskedLM.from_pretrained(base_pretrained_model)
    
    # must implement
    model.resize_token_embeddings(len(tokenizer))
    
    topic_sent, masked_sent = load_preprocessed(filename)
    print('after load data')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)
    
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=lr_rate)
    loss_record = []
    test_loss_record = []
    
    for epoch in range(epochs):
        train_masked, test_masked, train_topic, test_topic = train_test_split(masked_sent,
                                                                              topic_sent, 
                                                                              test_size=0.1, 
                                                                              random_state=42)
        train_dataloader = gen_data_loader(train_masked, train_topic, tokenizer)
        test_dataloader = gen_data_loader(test_masked, test_topic, tokenizer)
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        loss_record.append([str(epoch), str(float(loss))])

        with torch.no_grad():
            loop = tqdm(test_dataloader, leave=True)

            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0] 
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item()) 
            test_loss_record.append(str(float(loss)))
        
        # save model
        temp_dir_path = dir_path.replace(f'{epochs}epochs', f'{epoch+1}epochs')
        model.save_pretrained(temp_dir_path)
        with open(f'{temp_dir_path}/spec.txt', 'w') as f:
            f.write(f'ython train_MLM_further.py -e {epochs} -f {filename} --save -n {args.n} -r {reserve} -m {mark} -lr {lr_rate}\n')
            f.write(f'tokenizer: {tokenizer_name}\n')
            f.write(f'pretrained model: {base_pretrained_model}\n')
            for loss, validate in zip(loss_record, test_loss_record):
                f.write('\t'.join(loss) + '\t' + validate + '\n') 
            
if __name__ == '__main__':
    main()