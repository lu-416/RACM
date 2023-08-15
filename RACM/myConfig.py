import os
import torch
import numpy as np
import random
import time

def parameter_opt(opt):
    # data parameter
    opt.dataset = ''
    opt.split_token = '<SPLIT>'
    opt.vocab_size = 50000
    opt.max_title_len = 20
    opt.max_text_len = 200
    opt.max_code_len = 400
    opt.max_tag_len = 5

    # model parameter
    opt.hidden_size = 128
    opt.emb_size = 100
    opt.bidirectional = True
    opt.num_layers = 1
    opt.linear_size = [256]
    opt.encoder = ''

    # training parameter
    opt.gpuid = 1
    opt.gpus = [1, 5, 6, 7]
    opt.drop_rate = 0.1
    opt.learning_rate = 0.001
    opt.epochs = 20
    opt.batch_size = 128
    opt.start_epoch = 1
    opt.seed = 10
    opt.early_stop_tolerance = 10
    opt.learning_rate_decay = 0.5
    opt.top_K_list = [1, 3, 5]
    opt.single_metric = False
    opt.k_flod = 10
    opt.round = 1
    opt.report_one_epoch = True
    opt.weight_decay = 0.0

    opt.bert_path = "/data/lusijin/unixcoder-base"
    opt.bert_learning_rate = (7e-5)/4
    opt.bert_adam_epsilon = 1e-8
    opt.bert_dropout = 0.5

    return opt  

def path_opt(opt):
    # path
    opt.data_path = '/data/'
    opt.txt_path = opt.data_path + opt.dataset
    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    opt.timemark = timemark
    opt.model_path = opt.data_path+'model/%s.%s'
    return opt

def random_seed_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    return opt

def gpu_opt(opt):
    if torch.cuda.is_available():
        opt.device = torch.device("cuda:%d" % opt.gpuid)
        torch.cuda.set_device(opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        print("CUDA is not available, fall back to CPU.")  
    return opt

def process_opt(opt):
    opt = path_opt(opt)
    opt.exp = opt.dataset
    opt = random_seed_opt(opt)
    opt = gpu_opt(opt)

    size_tag = ".emb{}".format(opt.emb_size) + ".hid{}".format(opt.hidden_size)
    opt.exp += '.seed{}'.format(opt.seed)
    opt.exp += size_tag
    opt.model_path = opt.model_path % (opt.exp, opt.timemark)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    print('Model_PATH : ' + opt.model_path)  
    
    return opt        
       
