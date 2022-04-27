import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, AdamW

class TopicDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

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
    parser.add_argument('-f', type=str) # training file name
    parser.add_argument('-g', type=str) # directoy
    parser.add_argument('-n', type=str, default='all')
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-5)
    args = parser.parse_args()
    
    epochs = args.e
    filename = args.f
    reserve = bool(args.r)
    lr_rate = args.lr
    
    dir_path = f'./model/{args.g}/{args.n}_{reserve}_{epochs}epochs'
    os.mkdir(dir_path)
    print('model will be saved in: ', dir_path)
        
    tokenizer_name = 'tokenizer_casedFalse'
        
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print("use tokenizer:", tokenizer_name)
    
    base_pretrained_model = 'bert-base-uncased'
    model = BertForMaskedLM.from_pretrained(base_pretrained_model)

    # must implement
    model.resize_token_embeddings(len(tokenizer))
    
    topic_sent, masked_sent = load_preprocessed(filename)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)

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
    
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=lr_rate)
    loss_record = []
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
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
    
    model.save_pretrained(dir_path)
    with open(f'{dir_path}/spec.txt', 'w') as f:
        f.write(f'python train_MLM.py -e {epochs} -g {args.g} -f {filename} -n {args.n} -r {args.r}\n')
        f.write(f'tokenizer: {tokenizer_name}\n')
        f.write(f'pretrained model: {base_pretrained_model}\n')
        for loss in loss_record:
            f.write('\t'.join(loss) + '\n')
            
            
if __name__ == '__main__':
    main()