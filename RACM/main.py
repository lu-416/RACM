from myConfig import parameter_opt
import argparse
from ra import ra


def main(opt):
    opt.bert_batch_size = 2
    opt.epochs = 3
    opt.dataset = 'RACM/codereview.txt'
    result = ra(opt)
    opt.dataset = 'RACM/askubuntu.txt'
    result = ra(opt)
    opt.dataset = 'RACM/stackoverflow.txt'
    result = ra(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    opt = parser.parse_args(args=[])
    opt = parameter_opt(opt)

    opt.gpuid = 0
    opt.gpus = [0, 1, 2, 3]
    opt.k_flod = 10
    opt.use_pretrained = False
    main(opt)