from myUtils import convert_time2str
from raModel import myLoss
import torch
import os
import time
import sys
import math
import matplotlib.pyplot as plt

def bertTrain(model, optimizer, train_data_loader, valid_data_loader, opt):
    print('======================  Start Training  =========================')
    num_stop_dropping = 0
    best_valid_loss = float('inf')
    t0 = time.time()
    print("\nEntering main training for %d epochs" % opt.epochs)
    train_epochs_loss = []
    valid_epochs_loss = []
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, opt, epoch, train_data_loader)
        model.eval()
        valid_loss = valid_one_epoch(valid_data_loader, model, opt)
        if valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
            sys.stdout.flush()
            best_valid_loss = valid_loss
            num_stop_dropping = 0

            check_pt_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.model-%s' %
                                               (epoch, valid_loss, convert_time2str(time.time() - t0)))
            torch.save(
                model.state_dict(),
                open(check_pt_model_path, 'wb')
            )
        else:
            sys.stdout.flush()
            num_stop_dropping += 1

        print(
            'training loss: %.3f; validation loss: %.3f; best validation loss: %.3f' % (
                train_loss, valid_loss, best_valid_loss))

        if num_stop_dropping >= opt.early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break

        train_epochs_loss.append(train_loss)
        valid_epochs_loss.append(valid_loss)


    plt.figure(figsize=(12, 4))
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()

    print('check_pt_model_path', check_pt_model_path)
    return check_pt_model_path


def train_one_epoch(model, optimizer, opt, epoch, train_data_loader):
    model.train()
    print("\nTraining epoch: {}/{}".format(epoch, opt.epochs))

    train_batch_num = len(train_data_loader)
    total_loss = 0
    for batch_i, batch in enumerate(train_data_loader):
        batch_loss = train_one_batch(batch, model, optimizer, opt, batch_i)
        total_loss += batch_loss
    current_train_loss = total_loss / train_batch_num
    return current_train_loss


def train_one_batch(batch, model, optimizer, opt, batch_i):
    # train for one batch
    title, text, code, tag = batch
    # move data to GPU if available
    title_mask = title['attention_mask'].to(opt.device)
    # cp(title_mask.shape, "title_mask")
    title_input_ids = title['input_ids'].squeeze(1).to(opt.device)
    # cp(title_input_ids.shape, "title_input_ids")
    text_mask = text['attention_mask'].to(opt.device)
    text_input_ids = text['input_ids'].squeeze(1).to(opt.device)
    code_mask = code['attention_mask'].to(opt.device)
    code_input_ids = code['input_ids'].squeeze(1).to(opt.device)
    tag = tag.to(opt.device)
    # model.train()
    optimizer.zero_grad()

    y_pred = model(title_input_ids, title_mask, text_input_ids, text_mask, code_input_ids, code_mask)
    loss = myLoss(y_pred, tag.float(), opt)


    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("text")
        print(text)
        print("tag")
        print(tag)
        raise ValueError("Loss is NaN")

    # back propagation on the normalized loss
    # loss.requires_grad_(True)
    loss.backward()
    optimizer.step()

    batch_loss = loss.item()
    return batch_loss


def valid_one_epoch(data_loader, model, opt):
    model.to(opt.device)
    model.eval()
    total_loss = 0
    valid_batch_num = len(data_loader)
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            title, text, code, tag = batch
            # move data to GPU if available
            title_mask = title['attention_mask'].to(opt.device)
            title_input_ids = title['input_ids'].squeeze(1).to(opt.device)
            text_mask = text['attention_mask'].to(opt.device)
            text_input_ids = text['input_ids'].squeeze(1).to(opt.device)
            code_mask = code['attention_mask'].to(opt.device)
            code_input_ids = code['input_ids'].squeeze(1).to(opt.device)
            tag = tag.to(opt.device)

            y_pred = model(title_input_ids, title_mask, text_input_ids, text_mask, code_input_ids, code_mask)
            loss = myLoss(y_pred, tag.float(), opt)
            total_loss += loss.item()

    valid_loss = total_loss / valid_batch_num
    return valid_loss
