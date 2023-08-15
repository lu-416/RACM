import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from transformers import AdamW
import torch.nn.functional as F
import numpy as np
import faiss

class ContextAwareAttention(nn.Module):
    def __init__(self, dropout_rate, dim_model, dim_context):
        super(ContextAwareAttention, self).__init__()

        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model,
                                                     num_heads=1,
                                                     dropout=self.dropout_rate,
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True)

        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)

        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)

    def forward(self, q, k, v, context):
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output


class MAF(nn.Module):
    def __init__(self, dim_model, dropout_rate):
        super(MAF, self).__init__()
        self.dropout_rate = dropout_rate

        self.title_context_transform = nn.Linear(20, 200, bias=False)
        self.code_context_transform = nn.Linear(300, 200, bias=False)

        self.title_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=dim_model,
                                                                dropout_rate=dropout_rate)
        self.code_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=dim_model,
                                                              dropout_rate=dropout_rate)
        self.title_gate = nn.Linear(2 * dim_model, dim_model)
        self.code_gate = nn.Linear(2 * dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self, text_input, title_context, code_context):
        # Audio as Context for Attention
        title_context = title_context.permute(0, 2, 1)
        title_context = self.title_context_transform(title_context)
        title_context = title_context.permute(0, 2, 1)

        title_out = self.title_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=title_context)

        # Video as Context for Attention
        code_context = code_context.permute(0, 2, 1)
        code_context = self.code_context_transform(code_context)
        code_context = code_context.permute(0, 2, 1)


        code_out = self.code_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=code_context)

        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.title_gate(torch.cat((title_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.code_gate(torch.cat((code_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input +
                                       weight_a * title_out +
                                       weight_v * code_out)

        return output


class BERTEncoder(PreTrainedModel):
    def __init__(self, config, bert, num_class):
        super().__init__(config)

        self.tbert = AutoModel.from_pretrained(bert)
        self.nbert = AutoModel.from_pretrained(bert)
        self.cbert = AutoModel.from_pretrained(bert)

        self.MAF = MAF(config.hidden_size, config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)
        self.output_layer3 = nn.Linear(config.hidden_size * 3, num_class)

        self.cls_token = np.load('/data/lusijin/ks/cls.npy')
        self.t_ids = np.load('/data/lusijin/ks/title_ids.npy')
        self.n_ids = np.load('/data/lusijin/ks/text_ids.npy')
        self.c_ids = np.load('/data/lusijin/ks/code_ids.npy')
        self.t_att = np.load('/data/lusijin/ks/title_att.npy')
        self.n_att = np.load('/data/lusijin/ks/text_att.npy')
        self.c_att = np.load('/data/lusijin/ks/code_att.npy')

        self.cls_t = self.cls_token.astype('float32')
        self.title_ids = self.t_ids.astype('long')
        self.text_ids = self.n_ids.astype('long')
        self.code_ids = self.c_ids.astype('long')
        self.title_att = self.t_att.astype('long')
        self.text_att = self.n_att.astype('long')
        self.code_att = self.c_att.astype('long')

        param = 'Flat'
        measure = faiss.METRIC_L2
        self.k = 1
        self.index = faiss.index_factory(config.hidden_size * 3, param, measure)
        self.index.add(self.cls_t)  # 向index中添加向量

        print('index is trained：', self.index.is_trained)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        t = self.tbert(title_ids, attention_mask=title_attention_mask)
        n = self.nbert(text_ids, attention_mask=text_attention_mask)
        c = self.cbert(code_ids, attention_mask=code_attention_mask)

        t_hidden = t[0]
        n_hidden = n[0]
        c_hidden = c[0]

        cls_t = t[1]
        cls_n = n[1]
        cls_c = c[1]

        cls = torch.cat((cls_t, cls_n), 1)
        cls = torch.cat((cls, cls_c), 1)

        t1_ids = []
        t1_att = []
        n1_ids = []
        n1_att = []
        c1_ids = []
        c1_att = []
        t2_ids = []
        t2_att = []
        n2_ids = []
        n2_att = []
        c2_ids = []
        c2_att = []
        for i in range(0, len(cls)):
            cls_cpu = cls[i: i + 1].data.cpu().float().numpy()
            _, I = self.index.search(cls_cpu, self.k)  # 返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离

            t1_ids.append(torch.tensor(self.title_ids[I[0][0]: I[0][0] + 1]).cuda(t_hidden.device))
            t1_att.append(torch.tensor(self.title_att[I[0][0]: I[0][0] + 1]).cuda(t_hidden.device))

            n1_ids.append(torch.tensor(self.text_ids[I[0][0]: I[0][0] + 1]).cuda(n_hidden.device))
            n1_att.append(torch.tensor(self.text_att[I[0][0]: I[0][0] + 1]).cuda(n_hidden.device))

            c1_ids.append(torch.tensor(self.code_ids[I[0][0]: I[0][0] + 1]).cuda(c_hidden.device))
            c1_att.append(torch.tensor(self.code_att[I[0][0]: I[0][0] + 1]).cuda(c_hidden.device))

            t2_ids.append(torch.tensor(self.title_ids[I[0][1]: I[0][1] + 1]).cuda(t_hidden.device))
            t2_att.append(torch.tensor(self.title_att[I[0][1]: I[0][1] + 1]).cuda(t_hidden.device))

            n2_ids.append(torch.tensor(self.text_ids[I[0][1]: I[0][1] + 1]).cuda(n_hidden.device))
            n2_att.append(torch.tensor(self.text_att[I[0][1]: I[0][1] + 1]).cuda(n_hidden.device))

            c2_ids.append(torch.tensor(self.code_ids[I[0][1]: I[0][1] + 1]).cuda(c_hidden.device))
            c2_att.append(torch.tensor(self.code_att[I[0][1]: I[0][1] + 1]).cuda(c_hidden.device))

        title1_ids = t1_ids[0]
        title1_att = t1_att[0]
        for i in range(1, len(t1_ids)):
            title1_ids = torch.cat((title1_ids, t1_ids[i]), 0)
        for i in range(1, len(t1_att)):
            title1_att = torch.cat((title1_att, t1_att[i]), 0)

        text1_ids = n1_ids[0]
        text1_att = n1_att[0]
        for i in range(1, len(n1_ids)):
            text1_ids = torch.cat((text1_ids, n1_ids[i]), 0)
        for i in range(1, len(n1_ids)):
            text1_att = torch.cat((text1_att, n1_att[i]), 0)

        code1_ids = c1_ids[0]
        code1_att = c1_att[0]
        for i in range(1, len(c1_ids)):
            code1_ids = torch.cat((code1_ids, c1_ids[i]), 0)
        for i in range(1, len(c1_att)):
            code1_att = torch.cat((code1_att, c1_att[i]), 0)

        title2_ids = t2_ids[0]
        title2_att = t2_att[0]
        for i in range(1, len(t2_ids)):
            title2_ids = torch.cat((title2_ids, t2_ids[i]), 0)
        for i in range(1, len(t2_att)):
            title2_att = torch.cat((title2_att, t2_att[i]), 0)

        text2_ids = n2_ids[0]
        text2_att = n2_att[0]
        for i in range(1, len(n2_ids)):
            text2_ids = torch.cat((text2_ids, n2_ids[i]), 0)
        for i in range(1, len(n2_att)):
            text2_att = torch.cat((text2_att, n2_att[i]), 0)

        code2_ids = c2_ids[0]
        code2_att = c2_att[0]
        for i in range(1, len(c2_ids)):
            code2_ids = torch.cat((code2_ids, c2_ids[i]), 0)
        for i in range(1, len(c2_att)):
            code2_att = torch.cat((code2_att, c2_att[i]), 0)

        t1_hidden = self.tbert(title1_ids, attention_mask=title1_att)[0]
        n1_hidden = self.nbert(text1_ids, attention_mask=text1_att)[0]
        c1_hidden = self.cbert(code1_ids, attention_mask=code1_att)[0]

        t2_hidden = self.tbert(title2_ids, attention_mask=title2_att)[0]
        n2_hidden = self.nbert(text2_ids, attention_mask=text2_att)[0]
        c2_hidden = self.cbert(code2_ids, attention_mask=code2_att)[0]

        x = self.MAF(n_hidden, t_hidden, c_hidden)
        x = x[:, -1, :]
        x = self.dropout(x)

        y = self.MAF(n1_hidden, t1_hidden, c1_hidden)
        y = y[:, -1, :]
        y = self.dropout(y)

        z = self.MAF(n2_hidden, t2_hidden, c2_hidden)
        z = z[:, -1, :]
        z = self.dropout(z)

        con = torch.cat((x, y), 1)
        con = torch.cat((con, z), 1)

        logits = self.output_layer3(con)

        return logits


def init_optimizers(model, opt):
    optimizer = AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.bert_learning_rate, eps=opt.bert_adam_epsilon)
    return optimizer

def myLoss(y_pred, y_true, opt):
    criteria = nn.BCEWithLogitsLoss()
    # cp(y_pred.shape, "y_pred")
    # cp(y_true.shape, "y_true")
    loss = criteria(y_pred, y_true)
    return loss


def myModelStat(model):
    print('===========================Model Para==================================')
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print('===========================Model Para==================================')


