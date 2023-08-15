import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from transformers import AdamW
import torch.nn.functional as F
import numpy as np
import faiss

class ClassifyHeader(nn.Module):
    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.title_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.code_pooler = AvgPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, title_hidden, text_hidden, code_hidden):
        pool_title_hidden = self.title_pooler(title_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        pool_code_hidden = self.code_pooler(code_hidden)

        # concatenates the given sequence of tensors in the given dimension
        concated_hidden = torch.cat((pool_title_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, pool_code_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        x = self.output_layer(x)
        return x

class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


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

        self.title_context_transform = nn.Linear(40, 400, bias=False)
        self.code_context_transform = nn.Linear(600, 400, bias=False)

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
        self.output_layer2 = nn.Linear(config.hidden_size * 2, num_class)
        self.output_layer3 = nn.Linear(config.hidden_size * 3, num_class)

        self.cls = ClassifyHeader(config, num_class=num_class)

        self.cls_token = np.load('/data/lusijin/ks1/cls.npy')
        self.t_ids = np.load('/data/lusijin/ks1/title_ids.npy')
        self.n_ids = np.load('/data/lusijin/ks1/text_ids.npy')
        self.c_ids = np.load('/data/lusijin/ks1/code_ids.npy')
        self.t_att = np.load('/data/lusijin/ks1/title_att.npy')
        self.n_att = np.load('/data/lusijin/ks1/text_att.npy')
        self.c_att = np.load('/data/lusijin/ks1/code_att.npy')

        self.cls_t = self.cls_token.astype('float32')
        self.title_ids = self.t_ids.astype('long')
        self.text_ids = self.n_ids.astype('long')
        self.code_ids = self.c_ids.astype('long')
        self.title_att = self.t_att.astype('long')
        self.text_att = self.n_att.astype('long')
        self.code_att = self.c_att.astype('long')

        param = 'Flat'
        measure = faiss.METRIC_L2
        # measure = faiss.METRIC_L1
        # measure = faiss.METRIC_Linf
        # measure = faiss.METRIC_Canberra
        # measure = faiss.METRIC_INNER_PRODUCT
        self.k = 2
        self.index = faiss.index_factory(config.hidden_size * 3, param, measure)
        self.index.add(self.cls_t)  # 向index中添加向量

        print('index is trained：', self.index.is_trained)
        print('ks length：', len(self.cls_t))


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

        cls_cpu = cls.data.cpu().float().numpy()
        _, I = self.index.search(cls_cpu, self.k)  # 返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离

        t1_hidden = self.tbert(torch.tensor(self.title_ids[I[0][0]: I[0][0] + 1]).cuda(t_hidden.device), attention_mask=torch.tensor(self.title_att[I[0][0]: I[0][0] + 1]).cuda(t_hidden.device))[0]
        n1_hidden = self.nbert(torch.tensor(self.text_ids[I[0][0]: I[0][0] + 1]).cuda(n_hidden.device), attention_mask=torch.tensor(self.text_att[I[0][0]: I[0][0] + 1]).cuda(n_hidden.device))[0]
        c1_hidden = self.cbert(torch.tensor(self.code_ids[I[0][0]: I[0][0] + 1]).cuda(c_hidden.device), attention_mask=torch.tensor(self.code_att[I[0][0]: I[0][0] + 1]).cuda(c_hidden.device))[0]

        # t2_hidden = self.tbert(torch.tensor(self.title_ids[I[0][1]: I[0][1] + 1]).cuda(t_hidden.device), attention_mask=torch.tensor(self.title_att[I[0][1]: I[0][1] + 1]).cuda(t_hidden.device))[0]
        # n2_hidden = self.nbert(torch.tensor(self.text_ids[I[0][1]: I[0][1] + 1]).cuda(n_hidden.device), attention_mask=torch.tensor(self.text_att[I[0][1]: I[0][1] + 1]).cuda(n_hidden.device))[0]
        # c2_hidden = self.cbert(torch.tensor(self.code_ids[I[0][1]: I[0][1] + 1]).cuda(c_hidden.device), attention_mask=torch.tensor(self.code_att[I[0][1]: I[0][1] + 1]).cuda(c_hidden.device))[0]

        # t3_hidden = self.tbert(torch.tensor(self.title_ids[I[0][2]: I[0][2] + 1]).cuda(t_hidden.device), attention_mask=torch.tensor(self.title_att[I[0][2]: I[0][2] + 1]).cuda(t_hidden.device))[0]
        # n3_hidden = self.nbert(torch.tensor(self.text_ids[I[0][2]: I[0][2] + 1]).cuda(n_hidden.device), attention_mask=torch.tensor(self.text_att[I[0][2]: I[0][2] + 1]).cuda(n_hidden.device))[0]
        # c3_hidden = self.cbert(torch.tensor(self.code_ids[I[0][2]: I[0][2] + 1]).cuda(c_hidden.device), attention_mask=torch.tensor(self.code_att[I[0][2]: I[0][2] + 1]).cuda(c_hidden.device))[0]

        # t4_hidden = self.tbert(torch.tensor(self.title_ids[I[0][3]: I[0][3] + 1]).cuda(t_hidden.device), attention_mask=torch.tensor(self.title_att[I[0][3]: I[0][3] + 1]).cuda(t_hidden.device))[0]
        # n4_hidden = self.nbert(torch.tensor(self.text_ids[I[0][3]: I[0][3] + 1]).cuda(n_hidden.device), attention_mask=torch.tensor(self.text_att[I[0][3]: I[0][3] + 1]).cuda(n_hidden.device))[0]
        # c4_hidden = self.cbert(torch.tensor(self.code_ids[I[0][3]: I[0][3] + 1]).cuda(c_hidden.device), attention_mask=torch.tensor(self.code_att[I[0][3]: I[0][3] + 1]).cuda(c_hidden.device))[0]

        t_hidden = torch.cat((t1_hidden, t_hidden), 1)
        n_hidden = torch.cat((n1_hidden, n_hidden), 1)
        c_hidden = torch.cat((c1_hidden, c_hidden), 1)
        # print(t_hidden.shape)

        x = self.MAF(n_hidden, t_hidden, c_hidden)
        x = x[:, -1, :]
        x = self.dropout(x)

        logits = self.output_layer(x)

        # logits = self.cls(title_hidden=t_hidden,
        #                   text_hidden=n_hidden, code_hidden=c_hidden)

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


