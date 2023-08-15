import torch.utils.data as data_utils
from collections import Counter
import torch
import random
import numpy as np
from nltk.tokenize import word_tokenize
from myConfig import process_opt, parameter_opt
import argparse

def dataProcess(opt):
    tokenized_pairs = txt2list(opt.txt_path, opt.split_token)
    train, valid, test = dataSplit(tokenized_pairs, opt.seed)
    wordtt2idx, idx2wordtt, wordte2idx, idx2wordte, wordco2idx, idx2wordco, tag2idx, idx2tag = build_vocab(train, opt.vocab_size)

    Xtt_train, Xte_train, Xco_train, y_train = build_dataset(train, wordtt2idx, wordte2idx, wordco2idx, tag2idx, opt.vocab_size,
                                                             opt.max_title_len, opt.max_text_len, opt.max_code_len)
    Xtt_valid, Xte_valid, Xco_valid, y_valid = build_dataset(valid, wordtt2idx, wordte2idx, wordco2idx, tag2idx, opt.vocab_size,
                                                             opt.max_title_len, opt.max_text_len, opt.max_code_len)
    Xtt_test, Xte_test, Xco_test, y_test = build_dataset(test, wordtt2idx, wordte2idx, wordco2idx, tag2idx, opt.vocab_size,
                                                         opt.max_title_len, opt.max_text_len, opt.max_code_len)
    opt.tag_num = len(tag2idx)  
    
    # Normal feature and label
    train_data = data_utils.TensorDataset(torch.from_numpy(Xtt_train).type(torch.LongTensor),
                                          torch.from_numpy(Xte_train).type(torch.LongTensor),
                                          torch.from_numpy(Xco_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train).type(torch.LongTensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(Xtt_valid).type(torch.LongTensor),
                                        torch.from_numpy(Xte_valid).type(torch.LongTensor),
                                        torch.from_numpy(Xco_valid).type(torch.LongTensor),
                                        torch.from_numpy(y_valid).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(Xtt_test).type(torch.LongTensor),
                                         torch.from_numpy(Xte_test).type(torch.LongTensor),
                                         torch.from_numpy(Xco_test).type(torch.LongTensor),
                                         torch.from_numpy(y_test).type(torch.LongTensor))
    
    train_loader = data_utils.DataLoader(train_data, opt.batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(val_data, opt.batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, opt.batch_size, drop_last=True)
  
    print("======================Data Load Done======================")
    
    return train_loader, val_loader, test_loader, opt


def txt2list(txt_path, split_token):
    tokenized_text = []
    tokenized_title = []
    tokenized_code = []
    tokenized_tag = []

    max_text_len = 0
    max_title_len = 0
    max_code_len = 0
    max_tag_len = 0
    f = open(txt_path, 'r')
    for line in f.readlines():
        lineVec = line.strip().split(split_token)
        title_line = lineVec[0]
        text_line = lineVec[1]
        code_line = lineVec[2]
        tag_line = lineVec[3]
        title_word_list = word_tokenize(title_line)
        text_word_list = word_tokenize(text_line)
        code_word_list = word_tokenize(code_line)
        tag_word_list = tag_line.strip().split(';')

        if len(title_word_list) > max_title_len:
            max_title_len = len(title_word_list)
        if len(text_word_list) > max_text_len:
            max_text_len = len(text_word_list)
        if len(code_word_list) > max_code_len:
            max_code_len = len(code_word_list)
        if len(tag_word_list) > max_tag_len:
            max_tag_len = len(tag_word_list)

        tokenized_title.append(title_word_list)
        tokenized_text.append(text_word_list)
        tokenized_code.append(code_word_list)
        tokenized_tag.append(tag_word_list)

    assert len(tokenized_text) == len(tokenized_tag), \
        'the number of records in source and target are not the same'
    
    print('max_tit_len', max_title_len)
    print('max_text_len', max_text_len)
    print('max_code_len', max_code_len)
    print('max_trg_len', max_tag_len)
    
    tokenized_pairs = list(zip(tokenized_title, tokenized_text, tokenized_code, tokenized_tag))
    print("Finish reading %d lines" % len(tokenized_text))
    return tokenized_pairs


def dataSplit(tokenized_pairs, random_seed):
    random.seed(random_seed)
    random.shuffle(tokenized_pairs)
    data_length = len(tokenized_pairs)
    train_length = int(data_length*.8)
    valid_length = int(data_length*.9)
    train, valid, test = tokenized_pairs[:train_length], tokenized_pairs[train_length:valid_length],\
                                     tokenized_pairs[valid_length:]  
    return train, valid, test


def build_vocab(tokenized_pairs, vocab_size):
    '''
    Build the vocabulary from the training (src, trg) pairs
    :param tokenized_src_trg_pairs: list of (src, trg) pairs
    :return: word2idx, idx2word, token_freq_counter
    '''
    # Build vocabulary from training src and trg
    print("Building vocabulary from training data")
    token_freq_counter_title = Counter()
    token_freq_counter_text = Counter()
    token_freq_counter_code = Counter()
    token_freq_counter_tag = Counter()

    for title_word_list, text_word_list, code_word_list, tag_word_list in tokenized_pairs:
        token_freq_counter_title.update(title_word_list)
        token_freq_counter_text.update(text_word_list)
        token_freq_counter_code.update(code_word_list)
        token_freq_counter_tag.update(tag_word_list)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<unk>']
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter_title:
            del token_freq_counter_title[s_t]
        if s_t in token_freq_counter_text:
            del token_freq_counter_text[s_t]
        if s_t in token_freq_counter_code:
            del token_freq_counter_code[s_t]

    wordtt2idx = dict()
    idx2wordtt = dict()
    wordte2idx = dict()
    idx2wordte = dict()
    wordco2idx = dict()
    idx2wordco = dict()

    for idx, word in enumerate(special_tokens):
        wordtt2idx[word] = idx
        idx2wordtt[idx] = word
        wordte2idx[word] = idx
        idx2wordte[idx] = word
        wordco2idx[word] = idx
        idx2wordco[idx] = word

    # title
    sorted_wordtt2idx = sorted(token_freq_counter_title.items(), key=lambda x: x[1], reverse=True)
    sorted_wordstt = [x[0] for x in sorted_wordtt2idx]

    for idx, wordtt in enumerate(sorted_wordstt):
        wordtt2idx[wordtt] = idx + num_special_tokens

    for idx, wordtt in enumerate(sorted_wordstt):
        idx2wordtt[idx + num_special_tokens] = wordtt

    # text
    sorted_wordte2idx = sorted(token_freq_counter_text.items(), key=lambda x: x[1], reverse=True)
    sorted_wordste = [x[0] for x in sorted_wordte2idx]

    for idx, wordte in enumerate(sorted_wordste):
        wordte2idx[wordte] = idx + num_special_tokens

    for idx, wordte in enumerate(sorted_wordste):
        idx2wordte[idx + num_special_tokens] = wordte

    # code
    sorted_wordco2idx = sorted(token_freq_counter_code.items(), key=lambda x: x[1], reverse=True)
    sorted_wordsco = [x[0] for x in sorted_wordco2idx]

    for idx, wordco in enumerate(sorted_wordsco):
        wordco2idx[wordco] = idx + num_special_tokens

    for idx, wordco in enumerate(sorted_wordsco):
        idx2wordco[idx + num_special_tokens] = wordco

    # tag
    tag2idx = dict()
    idx2tag = dict()

    sorted_tag2idx = sorted(token_freq_counter_tag.items(), key=lambda x: x[1], reverse=True)
    sorted_tags = [x[0] for x in sorted_tag2idx]

    for idx, tag in enumerate(sorted_tags):
        tag2idx[tag] = idx

    for idx, tag in enumerate(sorted_tags):
        idx2tag[idx] = tag       
        
    print("Total title vocab_size: %d, text vocab_size: %d, code vocab_size: %d, predefined vocab_size: %d" % (len(wordtt2idx), len(wordte2idx), len(wordco2idx), vocab_size))
    print("Total tag_size: %d" % len(tag2idx))
    
    return wordtt2idx, idx2wordtt, wordte2idx, idx2wordte, wordco2idx, idx2wordco, tag2idx, idx2tag

def build_dataset(tokenized_pairs, wordtt2idx, wordte2idx, wordco2idx, tag2idx, vocab_size, max_title_len, max_text_len, max_code_len):
    '''
    word2id + padding + onehot
    '''
    title = []
    text = []
    code = []
    tag = []
    for idx, (title_src, text_src, code_src, tag_src) in enumerate(tokenized_pairs):
        title_l = [wordtt2idx[w] if w in wordtt2idx and wordtt2idx[w] < vocab_size
               else wordtt2idx['<unk>'] for w in title_src]
        text_l = [wordte2idx[w] if w in wordte2idx and wordte2idx[w] < vocab_size
               else wordte2idx['<unk>'] for w in text_src]
        code_l = [wordco2idx[w] if w in wordco2idx and wordco2idx[w] < vocab_size
                else wordco2idx['<unk>'] for w in code_src]
        tag_l = [tag2idx[w] for w in tag_src if w in tag2idx]
        title.append(title_l)
        text.append(text_l)
        code.append(code_l)
        tag.append(tag_l)
    title = padding(title, max_title_len, wordtt2idx)
    text = padding(text, max_text_len, wordte2idx)
    code = padding(code, max_code_len, wordco2idx)
    tag = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in tag]
    title = np.array(title)
    text = np.array(text)
    code = np.array(code)
    tag = np.array(tag)
    print('title.shape', title.shape)
    print('text.shape', text.shape)
    print('code.shape', code.shape)
    print('tag.shape', tag.shape)
    return title, text, code, tag

def padding(input_list, max_seq_len, word2idx):
    padded_batch = word2idx['<pad>'] * np.ones((len(input_list), max_seq_len), dtype=np.int64)
    for j in range(len(input_list)):
        current_len = len(input_list[j])
        if current_len <= max_seq_len:
            padded_batch[j][:current_len] = input_list[j]
        else:
            padded_batch[j] = input_list[j][:max_seq_len]
    return padded_batch


def encode_one_hot(inst, vocab_size, label_from):
    '''
    one hot for a value x, int, x>=1
    '''
    one_hots = np.zeros(vocab_size, dtype=np.float32)
    for value in inst:
        one_hots[value-label_from]=1
    return one_hots

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    opt = parser.parse_args(args=[])
    opt = parameter_opt(opt)
    opt = process_opt(opt)

    dataProcess(opt)
