import torch
from transformers import AdamW
from utils import modeling

def model_setup(num_labels, model_select, device, config, gen, dropout, dropoutrest):
    
    print("current dropout is: ", dropout, dropoutrest)
    if model_select == 'Bert':
        print(100*"#")
        print("using Bert")
        print(100*"#")
        model = modeling.bert_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    if model_select == 'Bert_mnli':
        print(100*"#")
        print("using Bert mnli")
        print(100*"#")
        model = modeling.bert_mnli_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    elif model_select == 'Bert_large_mnli':
        print(100*"#")
        print("using Bert large mnli")
        print(100*"#")
        model = modeling.bert_mnli_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    elif model_select == 'XLNet_base':
        print(100*"#")
        print("using XLNet_base base with class labels:",num_labels)
        print(100*"#")
        # model = modeling.roberta_large_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        model = modeling.xlnet_base_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)


        for n, p in model.named_parameters():
            print("n:",n,p.size())
        # print(bk)
        for n, p in model.named_parameters():
            if "bert.word_embedding" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'XLNet_base_mnli':
        print(100*"#")
        print("using XLNet_base mnli with class labels:",num_labels)
        print(100*"#")
        # model = modeling.roberta_large_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        model = modeling.xlnet_base_mnli_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)


        for n, p in model.named_parameters():
            print("n:",n,p.size())
        # print(bk)
        for n, p in model.named_parameters():
            if "bert.word_embedding" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'Bart':
        print(100*"#")
        print("using Bart")
        print(100*"#")
        model = modeling.bart_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    elif model_select == 'RoBERTa_large':
        print(100*"#")
        print("using RoBERTa large with class labels:",num_labels)
        print(100*"#")
        model = modeling.roberta_large_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)


        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'RoBERTa_large_mnli':
        print(100*"#")
        print("using RoBERTa large mnli with class labels:",num_labels)
        print(100*"#")
        model = modeling.roberta_large_mnli_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)


        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'RoBERTa_base':
        print(100*"#")
        print("using RoBERTa base with class labels:",num_labels)
        print(100*"#")
        # model = modeling.roberta_large_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        model = modeling.roberta_base_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)


        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'RoBERTa_base_mnli':
        print(100*"#")
        print("using RoBERTa base MNLI with class labels:",num_labels)
        print(100*"#")
        # model = modeling.roberta_large_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        model = modeling.roberta_base_mnli_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)


        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'Bart_encoder':
        print(100*"#")
        print("using Bart large mnli encoder")
        print(100*"#")
        model = modeling.bart_mnli_encoder_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    elif model_select == 'Bart_encoder_multi_nli':
        print(100*"#")
        print("using Bart large mnli encoder")
        print(100*"#")
        model = modeling.bart_multi_nli_encoder_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
    elif model_select == 'Bart_mnli_full':
        print(100*"#")
        print("using Bart large mnli full model")
        print(100*"#")
        model = modeling.bart_mnli_full_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "bart.model.shared.weight" in n \
                or "bart.model.decoder.embed" in n\
                or "bart.model.encoder.embed" in n:
                p.requires_grad = False
                print("require grad false n:",n)
        # print(bk)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.model')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.classification')], 'lr': float(config['fc_lr'])},            
            ]    
    elif model_select == 'XLM_RoBERTa_base':
        print(100*"#")
        print("using XLM_RoBERTa_base with class labels:",num_labels)
        print(100*"#")
        model = modeling.xlmroberta_base_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False
                # print()
                # print(bk)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}            
            ]
    elif model_select == 'mBert_base':
        print(100*"#")
        print("using mBert")
        print(100*"#")
        model = modeling.mbert_base_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]    
    elif model_select == 'mBart_large':
        print(100*"#")
        print("using mBart_large with class labels:",num_labels)
        print(100*"#")
        model = modeling.mbart_large_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        for n, p in model.named_parameters():
            print("n:",n,p.size())
        
        for n, p in model.named_parameters():
            if "bart.model.shared.weight" in n \
                or "bart.model.decoder.embed" in n\
                or "bart.model.encoder.embed" in n:
                p.requires_grad = False
                print("require grad false n:",n)
        # print(bk)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.model')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.classification')], 'lr': float(config['fc_lr'])},            
            ]    
        # print(100*"#")
        # for n, p in model.named_parameters():
        #     print("n:",n,p.size())
        # print(100*"#")
        # tmp = [n for n, p in model.named_parameters() if n.startswith('bart.model')]
        # for x in tmp:
        #     print("params:",x)
        # print(100*"#")
        # tmp2 = [n for n, p in model.named_parameters() if n.startswith('bart.classification')]
        # for x in tmp2:
        #     print("params2:",x)
        # print(100*"#")
        # print(bk) 
    elif model_select == 'mT5_base':
        print(100*"#")
        print("using mT5_base with class labels:",num_labels)
        print(100*"#")
        model = modeling.mt5_base_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)

        for n, p in model.named_parameters():
            print("n:",n,p.size())
        # print(bk)
        for n, p in model.named_parameters():
            if "mT5.shared.weight" in n:
                p.requires_grad = False
                print("require grad false n:",n)
        # print(bk)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('mT5.encoder')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]            
    optimizer = AdamW(optimizer_grouped_parameters)
    
    return model, optimizer


def model_preds(loader, model, device, loss_function,model_name):
    
    preds = [] 
    valtest_loss = []   
    for b_id, sample_batch in enumerate(loader):
        dict_batch = batch_fn(sample_batch,model_name)
        inputs = {k: v.to(device) for k, v in dict_batch.items()}
        outputs = model(**inputs)
        preds.append(outputs)

        loss = loss_function(outputs, inputs['gt_label'])
        valtest_loss.append(loss.item())
        
    return torch.cat(preds, 0), valtest_loss


def batch_fn(sample_batch,model_name):
    
    dict_batch = {}
    dict_batch['input_ids'] = sample_batch[0]
    dict_batch['attention_mask'] = sample_batch[1]
    dict_batch['gt_label'] = sample_batch[-1]
    if model_name=='Bert' or model_name=='mBert_base' or model_name=='Bert_mnli':
        dict_batch['token_type_ids'] = sample_batch[-2]
    
    return dict_batch