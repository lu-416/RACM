import random

from torch import nn
from transformers import BertConfig
from myUtils import time_since
from raDataset import dataProcess
from raModel import BERTEncoder, init_optimizers
from raTrain import bertTrain
from raTest import bertTest
from myConfig import process_opt
import time

def ra(opt):
    k_fold_result = []
    for i, random_seed in enumerate(random.sample(range(0, 99999), opt.k_flod)):
        opt.seed = 11267
        print('==================================Experiment %d================================' % i)

        # opt
        opt = process_opt(opt)

        # Dataset
        start_time = time.time()
        train_loader, val_loader, test_loader, opt = dataProcess(opt)
        load_data_time = time_since(start_time)
        print('Time for loading the data: %.1f' % load_data_time)

        # Model
        start_time = time.time()
        model = BERTEncoder(BertConfig(), opt.bert_path, opt.tag_num).to(opt.device)
        model = nn.DataParallel(model, device_ids=opt.gpus, output_device=opt.gpuid)
        optimizer = init_optimizers(model, opt)

        # Train
        if opt.use_pretrained != True:
            check_pt_model_path = bertTrain(model, optimizer,
                                            train_loader, val_loader, opt)
            training_time = time_since(start_time)
            print('Time for training: %.1f' % training_time)
        else:
            check_pt_model_path = ''

        # Test
        start_time = time.time()
        result = bertTest(model, test_loader, check_pt_model_path, opt)
        test_time = time_since(start_time)
        print('Time for testing: %.1f' % test_time)
        k_fold_result.append(result)

    return k_fold_result


