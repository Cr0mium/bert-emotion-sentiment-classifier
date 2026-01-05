import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class emotionDataset(Dataset):
    def __init__(self,data,key,tokenizer,max_length=128):
        self.text=data[key]['text']
        self.label=data[key]['label']
        self.max_length=max_length
        self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):

        encodings=self.tokenizer(
            self.text[index],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids':encodings['input_ids'].squeeze(0),
            'attention_mask':encodings['attention_mask'].squeeze(0),
            'labels':torch.tensor(self.label[index])
        }
    

def get_dataLoaders(data,batch_size=16):
    tokenizer= BertTokenizer.from_pretrained("bert-base-uncased")

    train_ds= emotionDataset(data,'train',tokenizer,128)
    validation_ds= emotionDataset(data,'validation',tokenizer,128)
    test_ds= emotionDataset(data,'test',tokenizer,128)

    train_dl= DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    validation_dl= DataLoader(train_ds,batch_size=batch_size)
    test_dl= DataLoader(train_ds,batch_size=batch_size)

    return train_dl,validation_dl, test_dl