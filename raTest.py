import numpy as np
import torch

def bertTest(model, test_data_loader, check_pt_model_path, opt):
    print("**************************************")
    print("**************************************")
    print("this is the final results")

    # load best model
    model.load_state_dict(torch.load(check_pt_model_path, map_location=opt.device))
    model.to(opt.device)
    model.eval()

    # test
    y_test = []
    y_pred = []
    with torch.no_grad():
        for batch_i, batch in enumerate(test_data_loader):
            title, text, code, tag = batch
            # move data to GPU if available
            title_mask = title['attention_mask'].to(opt.device)
            title_input_ids = title['input_ids'].squeeze(1).to(opt.device)
            text_mask = text['attention_mask'].to(opt.device)
            text_input_ids = text['input_ids'].squeeze(1).to(opt.device)
            code_mask = code['attention_mask'].to(opt.device)
            code_input_ids = code['input_ids'].squeeze(1).to(opt.device)
            tag = tag.to(opt.device)

            pred = model(title_input_ids, title_mask, text_input_ids, text_mask, code_input_ids, code_mask)

            tag_cpu = tag.data.cpu().float().numpy()
            pred_cpu = pred.data.cpu().numpy()
            pred_cpu = np.exp(pred_cpu)

            y_test.append(tag_cpu)
            y_pred.append(pred_cpu)
    result = evaluate(y_test, y_pred, opt)
    return result


def evaluate(y_real, y_pred, opt):
    y_real = np.array(y_real)
    y_real = np.reshape(y_real, (-1, y_real.shape[-1]))
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred, (-1, y_pred.shape[-1]))
    if opt.single_metric:
        top_K = 5
        precision, recall, f1 = evaluator(y_real, y_pred, top_K)
        print('pre@%d,re@%d,f1@%d' % (top_K, top_K, top_K))
        print(round(precision, opt.round), round(recall, opt.round), round(f1, opt.round))
        return f1
    else:
        result = []
        for top_K in opt.top_K_list:
            precision, recall, f1 = evaluator(y_real, y_pred, top_K)
            print('pre@%d,re@%d,f1@%d' % (top_K, top_K, top_K))
            print(round(precision, opt.round), round(recall, opt.round), round(f1, opt.round))
            result.append([precision, recall, f1])
        return result


def evaluator(y_true, y_pred, top_K):
    precision_K = []
    recall_K = []
    for i in range(y_pred.shape[0]):
        if np.sum(y_true[i, :]) == 0:
            continue
        top_indices = y_pred[i].argsort()[-top_K:]
        p = np.sum(y_true[i, top_indices]) / top_K
        r = np.sum(y_true[i, top_indices]) / np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
    precision = np.mean(np.array(precision_K)) * 100
    recall = np.mean(np.array(recall_K)) * 100
    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 3), round(recall, 3), round(f1, 3)