import os
os.environ["CUDA_VISIBLE_DEVICES"] ="3"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import gc
import gspread
import utils.preprocessing as pp
import utils.data_helper as dh
# import utils.data_helper_spuriosity as dh


from transformers import AdamW
from utils import modeling, evaluation, model_utils
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,classification_report

from torch.utils.tensorboard import SummaryWriter   
from pytorchtools import EarlyStopping



# CUDA_VISIBLE_DEVICES=0

def compute_performance(preds,y,trainvaltest,step,args,seed, model_select):
    print("preds:",preds,preds.size())
    print("y:",y,y.size())
    preds_np = preds.cpu().numpy()
    # if model_select=='Bart_encoder_multi_nli':
        # print(200*"#")
        # print("swap class rows for:",model_select)
        # print(200*"#")
        # preds_np = preds_np[:,[2,0,1]]
    preds_np = np.argmax(preds_np, axis=1)
    y_train2_np = y.cpu().numpy()

    #检查是否是在mix positive prompt和negative prompt
    if args['is_pos_neg_prompts_mixed']=='y' and trainvaltest!='test1' and trainvaltest!='test2':
        #load 保存的两种不同prompt的index，也就是哪个instance用了哪个prompt
        if 'train' in trainvaltest:
            trainvaltest2='train'
        elif 'val' in trainvaltest:
            trainvaltest2='val'
        elif 'test'==trainvaltest:
            trainvaltest2='test'

        def extract_path(s):
            # 查找最后一个斜杠的位置
            last_slash_index = s.rfind('/')
            # 如果没有找到斜杠，返回原字符串
            if last_slash_index == -1:
                return s
            # 返回到最后一个斜杠的子字符串，包括斜杠
            return s[:last_slash_index+1]

        path = extract_path(args['train_data'])+'prompt_indices_'+trainvaltest2+'.csv'
        indices_df = pd.read_csv(path)
        print(path,' , loaded!')
        indices_np = indices_df.to_numpy()
        assert len(indices_df)==len(preds_np)
        for tmp in range(len(preds_np)):
            if int(indices_np[tmp])>100:
                print("tmp before:",tmp,preds_np[tmp],y_train2_np[tmp])
                if preds_np[tmp]==1:
                    preds_np[tmp]=0
                elif preds_np[tmp]==0: 
                    preds_np[tmp]=1


                if y_train2_np[tmp]==1:
                    y_train2_np[tmp]=0
                elif y_train2_np[tmp]==0: 
                    y_train2_np[tmp]=1 

                # print("tmp after:",tmp,preds_np[tmp],y_train2_np[tmp])
                # print(20*"-")

    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')

    print("-------------------------------------------------------------------------------------")
    print(trainvaltest + " classification_report for step: {}".format(step))
    target_names = ['Against', 'Favor', 'neutral']
    print(classification_report(y_train2_np, preds_np, target_names = target_names, digits = 4))
    ###############################################################################################
    ################            Precision, recall, F1 to csv                     ##################
    ###############################################################################################
    # y_true = out_label_ids
    # y_pred = preds
    results_twoClass = precision_recall_fscore_support(y_train2_np, preds_np, average=None)
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')
    print("results_weighted:",results_weighted)
    result_overall = [results_weighted[0],results_weighted[1],results_weighted[2]]
    result_against = [results_twoClass[0][0],results_twoClass[1][0],results_twoClass[2][0]]
    result_favor = [results_twoClass[0][1],results_twoClass[1][1],results_twoClass[2][1]]
    result_neutral = [results_twoClass[0][2],results_twoClass[1][2],results_twoClass[2][2]]

    print("result_overall:",result_overall)
    print("result_favor:",result_favor)
    print("result_against:",result_against)
    print("result_neutral:",result_neutral)

    result_id = ['train', args['gen'], step, seed, args['dropout'],args['dropoutrest']]
    result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall
    result_one_sample = [result_one_sample]
    print("result_one_sample:",result_one_sample)

    # if results_weighted[2]>best_train_f1macro:
    #     best_train_f1macro = results_weighted[2]
    #     best_train_result = result_one_sample

    results_df = pd.DataFrame(result_one_sample)    
    print("results_df are:",results_df.head())
    results_df.to_csv('./results_'+trainvaltest+'_df.csv',index=False, mode='a', header=False)    
    print('./results_'+trainvaltest+'_df.csv save, done!')
    print("----------------------------------------------------------------------------")

    return results_weighted[2],result_one_sample

def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-g', '--gen', help='Generation number of student model', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-d', '--dropout', help='Dropout rate', required=False)
    parser.add_argument('-d2', '--dropoutrest', help='Dropout rate for rest generations', required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-kg', '--kg_data', help='Name of the kg test data file', default=None, required=False)
    parser.add_argument('-kg2', '--kg_data2', help='Name of the kg test data file', default=None, required=False)
    parser.add_argument('-clipgrad', '--clipgradient', type=str, default='True', help='whether clip gradient when over 2', required=False)
    parser.add_argument('-step', '--savestep', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-p', '--percent', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-es_step', '--earlystopping_step', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-mode', '--mode', type=str, default='train_en_test_zh', help='language for trainval, and test', required=True)
    parser.add_argument('-prompt_index', '--prompt_index', type=int, default=1, help='index to make nounphrase targets to a sentence', required=True)
    parser.add_argument('-is_pos_neg_prompts_mixed', '--is_pos_neg_prompts_mixed', type=str, default='n', help='if y, means you are mixing a positive prompt with a negative prompt', required=False)

    args = vars(parser.parse_args())


    print(200*"!")
    print("args['is_pos_neg_prompts_mixed']:",args['is_pos_neg_prompts_mixed'])
    print("args:",args)
    # print("args['config_file']['bert_lr']:",args['config_file']['bert_lr'])

    print(200*"!")
    # writer = SummaryWriter('./tensorboard/')

    sheet_num = 4  # Google sheet number
    num_labels = 3  # Favor, Against and None
#     random_seeds = [0,1,2,3,4,42]
    random_seeds = []
    random_seeds.append(int(args['seed']))
    
    # create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1,**data2}
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    model_select = config['model_select']

    print(200*"=")
    print("config['bert_lr']:",config['bert_lr'])
    print("config['fc_lr']:",config['fc_lr'])
    print("config['batch_size']:",config['batch_size'])
    print(200*"=")
    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    best_result, best_against, best_favor, best_val, best_val_against, best_val_favor,  = [], [], [], [], [], []
    for seed in random_seeds:    
        print("current random seed: ", seed)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'train')
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'val')
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        x_train, y_train, x_train_target = pp.clean_all(args['train_data'], norm_dict,args['mode'],'train',args['prompt_index'])
        x_val, y_val, x_val_target = pp.clean_all(args['dev_data'], norm_dict,args['mode'],'val',args['prompt_index'])
        x_test, y_test, x_test_target = pp.clean_all(args['test_data'], norm_dict,args['mode'],'test',args['prompt_index'])

        x_test_kg, y_test_kg, x_test_target_kg = pp.clean_all(args['kg_data'], norm_dict,args['mode'],'test',args['prompt_index'])
        x_test_kg2, y_test_kg2, x_test_target_kg2 = pp.clean_all(args['kg_data2'], norm_dict,args['mode'],'test',args['prompt_index'])

        x_train_all = [x_train,y_train,x_train_target]
        x_val_all = [x_val,y_val,x_val_target]
        x_test_all = [x_test,y_test,x_test_target]
        x_test_kg_all = [x_test_kg,y_test_kg,x_test_target_kg]
        x_test_kg_all2 = [x_test_kg2,y_test_kg2,x_test_target_kg2]

        if int(args['gen']) >= 1:
            print("Current generation is: ", args['gen'])
            x_train_all = [a+b for a,b in zip(x_train_all, x_test_kg_all)]
        print(x_test_all[0][0], x_test_all[1][0], x_test_all[2][0])

        # prepare for model
        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, x_test_kg_all, x_test_kg_all2, model_select, config)
        trainloader, valloader, testloader, trainloader2, kg_testloader1, kg_testloader2 = loader[0], loader[1], loader[2], loader[3], loader[4], loader[5]
        y_train, y_val, y_test, y_train2, y_test1, y_test2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3], gt_label[4], gt_label[5]
        y_val, y_test, y_train2, y_test1, y_test2 = y_val.to(device), y_test.to(device), y_train2.to(device), y_test1.to(device), y_test2.to(device)
        
        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, int(args['gen']), float(args['dropout']),float(args['dropoutrest']))
        loss_function = nn.CrossEntropyLoss()
        sum_loss = []
        val_f1_average, val_f1_against, val_f1_favor = [], [], []
        test_f1_average, test_f1_against, test_f1_favor, test_kg = [], [], [], []

        # early stopping


        es_intermediate_step = len(trainloader)//args['savestep']
        patience = args['earlystopping_step']   # the number of iterations that loss does not further decrease
        # patience = es_intermediate_step   # the number of iterations that loss does not further decrease        
        early_stopping = EarlyStopping(patience, args['seed'], verbose=True)
        print(100*"#")
        # print("len(trainloader):",len(trainloader))
        # print("args['savestep']:",args['savestep'])
        print("early stopping occurs when the loss does not decrease after {} steps.".format(patience))
        print(100*"#")
        # print(bk)
        # init best val/test results
        best_train_f1macro = 0
        best_train_result = []
        best_val_f1macro = 0
        best_val_result = []
        best_test_f1macro = 0
        best_test_result = []

        best_val_loss = 100000
        best_val_loss_result = []
        best_test_loss = 100000
        best_test_loss_result = []
        # start training
        print(100*"#")
        print("clipgradient:",args['clipgradient']=='True')
        print(100*"#")

        # model.eval()
        # with torch.no_grad():
        #     preds, loss_train = model_utils.model_preds(trainloader, model, device, loss_function)
        #     train_writer.add_scalar('loss', sum(loss_train) / len(loss_train), 0)
        #     preds, loss_val = model_utils.model_preds(valloader, model, device, loss_function)
        #     val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), 0)
        #     preds, loss_test = model_utils.model_preds(testloader, model, device, loss_function)
        #     test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), 0)
        step = 0
        # start training
        for epoch in range(0, int(config['total_epochs'])):
            print('Epoch:', epoch)
            train_loss = []  
            model.train()
            # #########################
            # with torch.no_grad():
            #     # preds_train, loss_train_inval_mode = model_utils.model_preds(trainloader2, model, device, loss_function,model_select)
            #     # preds_val, loss_val = model_utils.model_preds(valloader, model, device, loss_function,model_select)
            #     preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function,model_select)
            #     # preds_test1, loss_test1 = model_utils.model_preds(kg_testloader1, model, device, loss_function,model_select)
            #     # preds_test2, loss_test2 = model_utils.model_preds(kg_testloader2, model, device, loss_function,model_select)

            #     # f1macro_train, result_one_sample_train = compute_performance(preds_train,y_train2,'training',step, args, seed, config['model_select'])
            #     # f1macro_val, result_one_sample_val = compute_performance(preds_val,y_val,'validation',step, args, seed, config['model_select'])
            #     f1macro_test, result_one_sample_test = compute_performance(preds_test,y_test,'test',step, args, seed, config['model_select'])
            #     # f1macro_test1, result_one_sample_test1 = compute_performance(preds_test1,y_test1,'test1',step, args, seed, config['model_select'])                        
            #     # f1macro_test2, result_one_sample_test2 = compute_performance(preds_test2,y_test2,'test2',step, args, seed, config['model_select'])
            #     print(bk)
            # #########################
            for b_id, sample_batch in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                dict_batch = model_utils.batch_fn(sample_batch,model_select)
                inputs = {k: v.to(device) for k, v in dict_batch.items()}
                outputs = model(**inputs)
                loss = loss_function(outputs, inputs['gt_label'])
                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), 2)
                if args['clipgradient']=='True':
                    nn.utils.clip_grad_norm_(model.parameters(), 2)

                optimizer.step()
                step+=1
                train_loss.append(loss.item())

                # print("len(trainloader):",len(trainloader))
                split_step = len(trainloader)//args['savestep']
                # print("savestep:",savestep)
                # print(bk)
                # if step%args['savestep']==0:
                if step%split_step==0:
                    model.eval()
                    with torch.no_grad():
                        preds_train, loss_train_inval_mode = model_utils.model_preds(trainloader2, model, device, loss_function,model_select)
                        preds_val, loss_val = model_utils.model_preds(valloader, model, device, loss_function,model_select)
                        preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function,model_select)
                        preds_test1, loss_test1 = model_utils.model_preds(kg_testloader1, model, device, loss_function,model_select)
                        preds_test2, loss_test2 = model_utils.model_preds(kg_testloader2, model, device, loss_function,model_select)

                        print(100*"#")
                        print("at step: {}".format(step))
                        print("train_loss",train_loss,len(train_loss), sum(train_loss)/len(train_loss))
                        print("loss_val",loss_val,len(loss_val), sum(loss_val) / len(loss_val))
                        print("loss_test",loss_test,len(loss_test), sum(loss_test) / len(loss_test))
                        print("loss_test1",loss_test1,len(loss_test1), sum(loss_test1) / len(loss_test1))
                        print("loss_test2",loss_test2,len(loss_test2), sum(loss_test2) / len(loss_test2))

                        # print(bk)

                        train_writer.add_scalar('loss', sum(train_loss)/len(train_loss), step)
                        val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), step)
                        test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), step)
                        test_writer.add_scalar('loss', sum(loss_test1) / len(loss_test1), step)
                        test_writer.add_scalar('loss', sum(loss_test2) / len(loss_test2), step)


                        f1macro_train, result_one_sample_train = compute_performance(preds_train,y_train2,'training',step, args, seed, config['model_select'])
                        f1macro_val, result_one_sample_val = compute_performance(preds_val,y_val,'validation',step, args, seed, config['model_select'])
                        f1macro_test, result_one_sample_test = compute_performance(preds_test,y_test,'test',step, args, seed, config['model_select'])
                        f1macro_test1, result_one_sample_test1 = compute_performance(preds_test1,y_test1,'test1',step, args, seed, config['model_select'])                        
                        f1macro_test2, result_one_sample_test2 = compute_performance(preds_test2,y_test2,'test2',step, args, seed, config['model_select'])

                        train_writer.add_scalar('f1macro', f1macro_train, step)
                        val_writer.add_scalar('f1macro', f1macro_val, step)
                        test_writer.add_scalar('f1macro', f1macro_test, step)
                        test_writer.add_scalar('f1macro1', f1macro_test1, step)
                        test_writer.add_scalar('f1macro2', f1macro_test2, step)


                        if f1macro_val>best_val_f1macro:
                            best_val_f1macro = f1macro_val
                            best_val_result = result_one_sample_val
                            print(100*"#")
                            print("best f1-macro validation updated at epoch :{}, to: {}".format(epoch, best_val_f1macro))
                            best_test_f1macro = f1macro_test
                            best_test_result = result_one_sample_test

                            print("best f1-macro test updated at epoch :{}, to: {}".format(epoch, best_test_f1macro))
                            print(100*"#")

                        avg_val_loss = sum(loss_val) / len(loss_val)
                        avg_test_loss = sum(loss_test) / len(loss_test)
                        avg_test_loss1 = sum(loss_test1) / len(loss_test1)
                        avg_test_loss2 = sum(loss_test2) / len(loss_test2)
                        if avg_val_loss<best_val_loss:
                            best_val_loss = avg_val_loss
                            best_val_loss_result = result_one_sample_val
                            print(100*"#")
                            print("best loss validation updated at epoch :{}, to: {}".format(epoch, best_val_loss))
                            best_test_loss = avg_test_loss
                            best_test_loss_result = result_one_sample_test

                            best_test_loss1 = avg_test_loss1
                            best_test_loss_result1 = result_one_sample_test1
                            print("best loss test updated at epoch :{}, to: {}".format(epoch, best_test_loss1))

                            best_test_loss2 = avg_test_loss2
                            best_test_loss_result2 = result_one_sample_test2
                            print("best loss test updated at epoch :{}, to: {}".format(epoch, best_test_loss2))
                            print(100*"#")
                        # early stopping
                        print("loss_val:",loss_val,"average is: ",sum(loss_val) / len(loss_val))
                        early_stopping(sum(loss_val) / len(loss_val), model)
                        if early_stopping.early_stop:
                            print(100*"!")
                            print("Early stopping occurs at step: {}, stop training.".format(step))
                            print(100*"!")
                            break
                    model.train()

            if early_stopping.early_stop:
                print(100*"!")
                print("Early stopping, training ends")
                print(100*"!")
                break

            sum_loss.append(sum(train_loss)/len(train_loss))
            print(sum_loss[epoch])


        #########################################################
        best_val_result[0][0]='best validation'        
        results_df = pd.DataFrame(best_val_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./results_validation_df.csv',index=False, mode='a', header=False)    
        print('./results_validation_df.csv save, done!')
        ###
        results_df = pd.DataFrame(best_val_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_results_validation_df.csv',index=False, mode='a', header=False)    
        print('./best_results_validation_df.csv save, done!')
        ###
        best_val_loss_result[0][0]='best validation' 
        results_df = pd.DataFrame(best_val_loss_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_validation_df.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_validation_df.csv save, done!')
        #########################################################
        best_test_result[0][0]='best test'
        results_df = pd.DataFrame(best_test_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./results_test_df.csv',index=False, mode='a', header=False)    
        print('./results_test_df.csv save, done!')
        ###
        results_df = pd.DataFrame(best_test_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_results_test_df.csv',index=False, mode='a', header=False)    
        print('./best_results_test_df.csv save, done!')
        ###
        best_test_loss_result[0][0]='best test'
        results_df = pd.DataFrame(best_test_loss_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_test_df.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_test_df.csv save, done!')
        ###
        best_test_loss_result1[0][0]='best test1'
        results_df = pd.DataFrame(best_test_loss_result1)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_test_df1.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_test_df1.csv save, done!')
        ###
        best_test_loss_result2[0][0]='best test2'
        results_df = pd.DataFrame(best_test_loss_result2)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_test_df2.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_test_df2.csv save, done!')
        #########################################################
        # model that performs best on the dev set is evaluated on the test set
        '''
        best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        best_against.append(test_f1_against[best_epoch])
        best_favor.append(test_f1_favor[best_epoch])
        best_result.append(test_f1_average[best_epoch])

        print("******************************************")
        print("dev results with seed {} on all epochs".format(seed))
        print(val_f1_average)
        best_val_against.append(val_f1_against[best_epoch])
        best_val_favor.append(val_f1_favor[best_epoch])
        best_val.append(val_f1_average[best_epoch])
        print("******************************************")
        print("test results with seed {} on all epochs".format(seed))
        print(test_f1_average)
        print("******************************************")
        print(max(best_result))
        print(best_result)
        '''


if __name__ == "__main__":
    run_classifier()
