import numpy as np
import torch.utils.data as data_utils
import torch
from transformers import AutoTokenizer
from myDataset import txt2list, dataSplit, build_vocab, encode_one_hot

def dataProcess(opt):
    tokenized_pairs = txt2list(opt.txt_path, opt.split_token)
    train, valid, test = dataSplit(tokenized_pairs, opt.seed)
    wordtt2idx, idx2wordtt, wordte2idx, idx2wordte, wordco2idx, idx2wordco, tag2idx, idx2tag = build_vocab(train, opt.vocab_size)

    opt.tag_num = len(tag2idx)

    tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)
    Xtt_train, Xte_train, Xco_train, y_train = build_bert_dataset(tokenizer, train, tag2idx)
    Xtt_valid, Xte_valid, Xco_valid, y_valid = build_bert_dataset(tokenizer, valid, tag2idx)
    Xtt_test, Xte_test, Xco_test, y_test = build_bert_dataset(tokenizer, test, tag2idx)

    train_data = Dataset(Xtt_train, Xte_train, Xco_train, y_train)
    val_data = Dataset(Xtt_valid, Xte_valid, Xco_valid, y_valid)
    test_data = Dataset(Xtt_test, Xte_test, Xco_test, y_test)

    train_loader = data_utils.DataLoader(train_data, opt.bert_batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(val_data, opt.bert_batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, opt.bert_batch_size, drop_last=True)

    print("======================Data Load Done======================")
    
    return train_loader, val_loader, test_loader, opt

def build_bert_dataset(tokenizer, tokenized_pairs, tag2idx):
    '''
    build train/valid/test dataset
    '''
    title = []
    text = []
    code = []
    tag = []
    for idx, (title_source, text_source, code_source, targets) in enumerate(tokenized_pairs):
        title_src = ' '.join(title_source)
        title_src = tokenizer(title_src, padding='max_length', max_length=20, truncation=True,
                              return_tensors="pt")
        text_src = ' '.join(text_source)
        text_src = tokenizer(text_src, padding='max_length', max_length=200, truncation=True,
                             return_tensors="pt")
        code_src = ' '.join(code_source)
        code_src = tokenizer(code_src, padding='max_length', max_length=300, truncation=True,
                             return_tensors="pt")
        trg = [tag2idx[w] for w in targets if w in tag2idx]
        title.append(title_src)
        text.append(text_src)
        code.append(code_src)
        tag.append(trg)

    tag = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in tag]
    tag = np.array(tag)
    return title, text, code, tag

class Dataset(torch.utils.data.Dataset):
    def __init__(self, titles, texts, codes, tags):
        self.titles = titles
        self.texts = texts
        self.codes = codes
        self.tags = tags

    def classes(self):
        return self.tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        batch_titles = self.titles[idx]
        batch_texts = self.texts[idx]
        batch_codes = self.codes[idx]
        batch_y = self.tags[idx]

        return batch_titles, batch_texts, batch_codes, batch_y
        