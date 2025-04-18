import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from transformers import RobertaModel, MT5EncoderModel, MT5Model,MT5ForConditionalGeneration,XLNetModel

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,MBartForSequenceClassification


# XLM RoBERTa
class xlmroberta_base_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(xlmroberta_base_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.roberta = RobertaModel.from_pretrained("xlm-roberta-base")
        # self.roberta = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
        # self.roberta = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
        
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(1024*2, 1024)
        # self.linear = nn.Linear(1024, 1024)
        self.linear = nn.Linear(768, 768)
        
        self.eos_token_id=2

        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk) 

        # self.out = nn.Linear(1024, num_labels)         
        self.out = nn.Linear(768, num_labels)         

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        

        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("last_hidden[0]:",last_hidden[0].size())
        # print("last_hidden[1]:",last_hidden[1].size())
        # print("last_hidden[2]:",last_hidden[2].size())

        # print(bk)
        cls_hidden = last_hidden[0][:, 0, :]

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


# mBERT
class mbert_base_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(mbert_base_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls_hidden = last_hidden[0][:, 0, :]

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        ###########################################################################################

        
        # print("last_hidden:",last_hidden[0].size())
        # print(bk)
        # ###########################################################################################
        # x_atten_masks[:,0] = 0 # [CLS] --> 0 

        # idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        # x_seg_ind = x_seg_ids * idx
        # x_att_ind = (x_atten_masks-x_seg_ids) * idx
        # indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        # indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        # for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
        #     seg[seg_id] = 0  # 2nd [SEP] --> 0 
        #     att[att_id:] = 0  # 1st [SEP] --> 0 
        
        # txt_l = x_atten_masks.sum(1).to('cuda')
        # topic_l = x_seg_ids.sum(1).to('cuda')
        # txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        # topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        # txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        # topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        # cat = torch.cat((txt_mean, topic_mean), dim=1)
        # query = self.dropout(cat)
        # linear = self.relu(self.linear(query))
        # out = self.out(linear)
        # ###########################################################################################

        
        return out
# mBART
class mbart_large_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(mbart_large_classifier, self).__init__()
        
        self.bart = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25", num_labels=num_labels)

        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
            # self.bart.classification_head.out_proj = nn.Linear(1024, 2) 
            # print("self.bart.classification_head.out_proj:",self.bart.classification_head.out_proj)
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk)            
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        out = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("out:",out)
        # print("out[0]:",out[0].size())
        # print(bk)
        return out[0]


# mT5_base
class mt5_base_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(mt5_base_classifier, self).__init__()
        
        self.mT5 = MT5EncoderModel.from_pretrained("google/mt5-base")
        # self.mT5 = MT5Model.from_pretrained("google/mt5-base")
        # self.mT5 = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")


        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
            # self.bart.classification_head.out_proj = nn.Linear(1024, 2) 
            # print("self.bart.classification_head.out_proj:",self.bart.classification_head.out_proj)
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk) 

        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        self.mT5.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.mT5.config.hidden_size, self.mT5.config.hidden_size)
        self.out = nn.Linear(self.mT5.config.hidden_size, num_labels)
        

    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        outputs = self.mT5(x_input_ids)
        hidden_state = outputs.last_hidden_state
        # print("hidden_state:",hidden_state.size())
        avg_hidden_state = torch.mean(hidden_state, 1, False)
        # print("avg_hidden_state:",avg_hidden_state.size())

        query = self.dropout(avg_hidden_state)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        # print("out:",out.size())
        # print(bk)
        return out

# BERT
class bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        ###########################################################################################
        #                                       用CLS token
        ###########################################################################################
        cls_hidden = last_hidden[0][:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        ###########################################################################################
        #                                       用 BERT joint
        ###########################################################################################
        # x_atten_masks[:,0] = 0 # [CLS] --> 0 

        # idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        # x_seg_ind = x_seg_ids * idx
        # x_att_ind = (x_atten_masks-x_seg_ids) * idx
        # indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        # indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        # for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
        #     seg[seg_id] = 0  # 2nd [SEP] --> 0 
        #     att[att_id:] = 0  # 1st [SEP] --> 0 
        
        # txt_l = x_atten_masks.sum(1).to('cuda')
        # topic_l = x_seg_ids.sum(1).to('cuda')
        # txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        # topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        # txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        # topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        # cat = torch.cat((txt_mean, topic_mean), dim=1)
        # query = self.dropout(cat)
        # linear = self.relu(self.linear(query))
        # out = self.out(linear)

        # if torch.isnan(out).any():

        #     print("txt_l:",txt_l)
        #     print("topic_l:",topic_l)
        #     print("txt_vec:",txt_vec)
        #     print("topic_vec:",topic_vec)
        #     print("torch.einsum('blh,bl->bh', last_hidden[0], txt_vec):",torch.einsum('blh,bl->bh', last_hidden[0], txt_vec))
        #     print("torch.einsum('blh,bl->bh', last_hidden[0], topic_vec):",torch.einsum('blh,bl->bh', last_hidden[0], topic_vec))

        #     for x in out:
        #         print("x:",x)

        #     for idx in range(x_seg_ids.size()[0]):
        #         print("idx:",idx)
        #         if idx==12:
        #             for x in x_seg_ids[idx]:
        #                 print("x:",x_seg_ids)
        #     print(bk)
        ###########################################################################################

        
        return out

# BERT
class bert_mnli_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bert_mnli_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("textattack/bert-base-uncased-MNLI")
        # for n, p in self.bert.named_parameters():
        #     print("self.bert n:",n,p.size())
        # print(bk)
        # self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls_hidden = last_hidden[0][:, 0, :]

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        # print("last_hidden[0]:",last_hidden[0].size())
        # print("last_hidden[1]:",last_hidden[1].size())
        # print("last_hidden[2]:",last_hidden[2].size())
        # print(bk)
        ###########################################################################################

        
        # print("last_hidden:",last_hidden[0].size())
        # print(bk)
        # ###########################################################################################
        # x_atten_masks[:,0] = 0 # [CLS] --> 0 

        # idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        # x_seg_ind = x_seg_ids * idx
        # x_att_ind = (x_atten_masks-x_seg_ids) * idx
        # indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        # indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        # for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
        #     seg[seg_id] = 0  # 2nd [SEP] --> 0 
        #     att[att_id:] = 0  # 1st [SEP] --> 0 
        
        # txt_l = x_atten_masks.sum(1).to('cuda')
        # topic_l = x_seg_ids.sum(1).to('cuda')
        # txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        # topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        # txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        # topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        # cat = torch.cat((txt_mean, topic_mean), dim=1)
        # query = self.dropout(cat)
        # linear = self.relu(self.linear(query))
        # out = self.out(linear)
        # ###########################################################################################

        
        return out

    
# BART
class Encoder(BartPretrainedModel):
    
    def __init__(self, config: BartConfig):
        
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class bart_mnli_encoder_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bart_mnli_encoder_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
        self.bart.pooler = None
        # self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)

        cls_hidden = last_hidden[0][:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        '''
        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        # print("x_input_ids:",x_input_ids,x_input_ids.size())
        # print("x_atten_masks:",x_atten_masks,x_atten_masks.size())

        # print("len(eos_token_ind):",len(eos_token_ind))
        # print("len(x_input_ids):",len(x_input_ids),3*len(x_input_ids))
        # print(bk)
        assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        '''
        return out


class bart_multi_nli_encoder_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bart_multi_nli_encoder_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        self.config = BartConfig.from_pretrained(hg_model_hub_name)
        self.bart = Encoder.from_pretrained(hg_model_hub_name)
        self.bart.pooler = None
        # self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)

        cls_hidden = last_hidden[0][:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        '''
        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        # print("x_input_ids:",x_input_ids,x_input_ids.size())
        # print("x_atten_masks:",x_atten_masks,x_atten_masks.size())

        # print("len(eos_token_ind):",len(eos_token_ind))
        # print("len(x_input_ids):",len(x_input_ids),3*len(x_input_ids))
        # print(bk)
        assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        '''
        return out

class bart_mnli_full_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bart_mnli_full_classifier, self).__init__()
        
        self.bart = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

        if num_labels==2:
            # fc_features = model.fc.in_features  
            # model.fc = nn.Linear(fc_features, 2) 
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
            # for n, p in model.named_parameters():
            self.bart.classification_head.out_proj = nn.Linear(1024, 2) 
            print("self.bart.classification_head.out_proj:",self.bart.classification_head.out_proj)
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk)            
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        out = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)
        
        return out[0]

# xlnet
class xlnet_base_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(xlnet_base_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        self.bert = XLNetModel.from_pretrained('xlnet-base-cased')
        # self.bert = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

        self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        # x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        # output = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)

        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        output = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("output:",output)
        # print(100*"#")
        ###########################################################################################
        #                                       用CLS token
        ###########################################################################################
        cls_hidden = output.last_hidden_state[:, 0, :]
        # print("output.last_hidden_state:",output.last_hidden_state.size())
        # print(100*"#")
        # print("cls_hidden:",cls_hidden.size())
        # print(100*"#")
        # print(bk)
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out
        
class xlnet_base_mnli_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(xlnet_base_mnli_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        self.bert = XLNetModel.from_pretrained('textattack/xlnet-base-cased-MNLI')
        # self.bert = XLNetModel.from_pretrained('TehranNLP/xlnet-base-cased-mnli')

        self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        # x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        # output = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)

        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        output = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("output:",output)
        # print(100*"#")
        ###########################################################################################
        #                                       用CLS token
        ###########################################################################################
        cls_hidden = output.last_hidden_state[:, 0, :]
        # print("output.last_hidden_state:",output.last_hidden_state.size())
        # print(100*"#")
        # print("cls_hidden:",cls_hidden.size())
        # print(100*"#")
        # print(bk)
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out    
class roberta_large_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(roberta_large_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.roberta = model = RobertaModel.from_pretrained('roberta-large')
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(1024*2, 1024)
        self.linear = nn.Linear(1024, 1024)
        
        self.eos_token_id=2

        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk) 

        self.out = nn.Linear(1024, num_labels)         

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        

        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("last_hidden[0]:",last_hidden[0],last_hidden[0].size())
        # cls_hidden = last_hidden[:][0][:]
        cls_hidden = last_hidden[0][:, 0, :]
        # print("cls_hidden:",cls_hidden,cls_hidden.size())
        # print("last_hidden[1]:",last_hidden[1],last_hidden[1].size())
        # print(bk)

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)



        # eos_token_ind = x_input_ids.eq(self.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        # assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
        # b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        # e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        # x_atten_clone = x_atten_masks.clone().detach()
        # for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
        #     att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
        #     att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        # txt_l = x_atten_masks.sum(1).to('cuda')
        # topic_l = x_atten_clone.sum(1).to('cuda')
        # txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        # topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        # txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        # topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        # cat = torch.cat((txt_mean, topic_mean), dim=1)
        # query = self.dropout(cat)
        # linear = self.relu(self.linear(query))
        # out = self.out(linear)
        
        return out

class roberta_large_mnli_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(roberta_large_mnli_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.roberta = model = RobertaModel.from_pretrained('roberta-large-mnli')
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(1024*2, 1024)
        self.linear = nn.Linear(1024, 1024)
        
        self.eos_token_id=2

        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk) 

        self.out = nn.Linear(1024, num_labels)         

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        

        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("last_hidden[0]:",last_hidden[0],last_hidden[0].size())
        # cls_hidden = last_hidden[:][0][:]
        cls_hidden = last_hidden[0][:, 0, :]
        # print("cls_hidden:",cls_hidden,cls_hidden.size())
        # print("last_hidden[1]:",last_hidden[1],last_hidden[1].size())
        # print(bk)

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out

class roberta_base_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(roberta_base_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.roberta = model = RobertaModel.from_pretrained('roberta-base')
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(1024*2, 1024)
        self.linear = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)
        
        self.eos_token_id=2

        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk) 

        self.out = nn.Linear(self.roberta.config.hidden_size, num_labels)         

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        

        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        # print("last_hidden[0]:",last_hidden[0],last_hidden[0].size())
        # cls_hidden = last_hidden[:][0][:]
        cls_hidden = last_hidden[0][:, 0, :]
        # print("cls_hidden:",cls_hidden,cls_hidden.size())
        # print("last_hidden[1]:",last_hidden[1],last_hidden[1].size())
        # print(bk)

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out

class roberta_base_mnli_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(roberta_base_mnli_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.roberta = model = RobertaModel.from_pretrained('textattack/roberta-base-MNLI')
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(1024*2, 1024)
        self.linear = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)
        
        self.eos_token_id=2

        if num_labels==2:
            print(200*"!")
            print("Working with a 2-class classification task")
            print(200*"!")
        elif num_labels==3:
            print(200*"!")
            print("Working with a 3-class classification task")
            print(200*"!")
        else:
            print("num of class incorrect!")
            print(bk) 

        self.out = nn.Linear(self.roberta.config.hidden_size, num_labels)         

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        

        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)

        cls_hidden = last_hidden[0][:, 0, :]


        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out

# class bart_classifier(nn.Module):

#     def __init__(self, num_labels, model_select, gen, dropout):

#         super(bart_classifier, self).__init__()
        
#         self.dropout = nn.Dropout(0.1) if gen==0 else nn.Dropout(dropout)
#         self.relu = nn.ReLU()
        
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
#         self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
#     def forward(self, **kwargs):
        
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)
        
#         hidden_states = last_hidden[0] 
#         eos_mask = x_input_ids.eq(self.config.eos_token_id)

#         if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
#             raise ValueError("All examples must have the same number of <eos> tokens.")
#         query = hidden_states[eos_mask,:].view(hidden_states.size(0), -1, hidden_states.size(-1))[:,-1,:]

#         query = self.dropout(query)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)
        
#         return out