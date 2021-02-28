#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:34:27 2020

@author: luokai
"""

import torch
from transformers import BertModel, BertTokenizer
import json
from utils import get_f1, print_result, load_pretrained_model, load_tokenizer
from net import Net
from data_generator import Data_generator
from calculate_loss import Calculate_loss


def train(epochs=1, batchSize=2, lr=0.0001, device='cuda:0', accumulate=True, a_step=16, load_saved=False, file_path='./saved_best.pt', use_dtp=False, pretrained_model='./bert_pretrain_model', tokenizer_model='bert-base-chinese', weighted_loss=False):
    device = device
    tokenizer = load_tokenizer(tokenizer_model)
    my_net = torch.load(file_path) if load_saved else Net(load_pretrained_model(pretrained_model))
    my_net.to(device, non_blocking=True)
    label_dict = dict()
    with open('./tianchi_datasets/label.json') as f:        # 获得三个语料集的标签
        for line in f:
            label_dict = json.loads(line)
            break
    label_weights_dict = dict()
    with open('./tianchi_datasets/label_weights.json') as f:    # 获取每个标签对应权重？
        for line in f:
            label_weights_dict = json.loads(line)
            break
    ocnli_train = dict()
    with open('./tianchi_datasets/OCNLI/train.json') as f:     #下面是三个语料集训练集和测试集的划分
        for line in f:
            ocnli_train = json.loads(line)
            break
    ocnli_dev = dict()
    with open('./tianchi_datasets/OCNLI/dev.json') as f:
        for line in f:
            ocnli_dev = json.loads(line)
            break
    ocemotion_train = dict()
    with open('./tianchi_datasets/OCEMOTION/train.json') as f:
        for line in f:
            ocemotion_train = json.loads(line)
            break
    ocemotion_dev = dict()
    with open('./tianchi_datasets/OCEMOTION/dev.json') as f:
        for line in f:
            ocemotion_dev = json.loads(line)
            break
    tnews_train = dict()
    with open('./tianchi_datasets/TNEWS/train.json') as f:
        for line in f:
            tnews_train = json.loads(line)
            break
    tnews_dev = dict()
    with open('./tianchi_datasets/TNEWS/dev.json') as f:
        for line in f:
            tnews_dev = json.loads(line)
            break
    train_data_generator = Data_generator(ocnli_train, ocemotion_train, tnews_train, label_dict, device, tokenizer)   # 封装训练集数据
    dev_data_generator = Data_generator(ocnli_dev, ocemotion_dev, tnews_dev, label_dict, device, tokenizer)           # 封装验证集数据
    tnews_weights = torch.tensor(label_weights_dict['TNEWS']).to(device, non_blocking=True)              # 获得每个语料标签的权重
    ocnli_weights = torch.tensor(label_weights_dict['OCNLI']).to(device, non_blocking=True)
    ocemotion_weights = torch.tensor(label_weights_dict['OCEMOTION']).to(device, non_blocking=True)
    loss_object = Calculate_loss(label_dict, weighted=weighted_loss, tnews_weights=tnews_weights, ocnli_weights=ocnli_weights, ocemotion_weights=ocemotion_weights)
    optimizer=torch.optim.Adam(my_net.parameters(), lr=lr)
    best_dev_f1 = 0.0    # check points 以 f1 score 衡量
    best_epoch = -1
    for epoch in range(epochs):
        my_net.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        train_ocnli_correct = 0
        train_ocemotion_correct = 0
        train_tnews_correct = 0
        train_ocnli_pred_list = []
        train_ocnli_gold_list = []
        train_ocemotion_pred_list = []
        train_ocemotion_gold_list = []
        train_tnews_pred_list = []
        train_tnews_gold_list = []
        cnt_train = 0
        while True:
            raw_data = train_data_generator.get_next_batch(batchSize)   # 三个语料集轮番进行训练的方式
            if raw_data == None:
                break
            data = dict()
            data['input_ids'] = raw_data['input_ids']
            data['token_type_ids'] = raw_data['token_type_ids']
            data['attention_mask'] = raw_data['attention_mask']
            data['ocnli_ids'] = raw_data['ocnli_ids']
            data['ocemotion_ids'] = raw_data['ocemotion_ids']
            data['tnews_ids'] = raw_data['tnews_ids']
            tnews_gold = raw_data['tnews_gold']
            ocnli_gold = raw_data['ocnli_gold']
            ocemotion_gold = raw_data['ocemotion_gold']
            if not accumulate:
                optimizer.zero_grad()
            ocnli_pred, ocemotion_pred, tnews_pred = my_net(**data)    # data是一个字典类型变量，用**输入可依次出传入
            if use_dtp:
                tnews_kpi = 0.1 if len(train_tnews_pred_list) == 0 else train_tnews_correct / len(train_tnews_pred_list)
                ocnli_kpi = 0.1 if len(train_ocnli_pred_list) == 0 else train_ocnli_correct / len(train_ocnli_pred_list)
                ocemotion_kpi = 0.1 if len(train_ocemotion_pred_list) == 0 else train_ocemotion_correct / len(train_ocemotion_pred_list)
                current_loss = loss_object.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold,
                                                   ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
            else:
                current_loss = loss_object.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)    #计算损失
            train_loss += current_loss.item()
            current_loss.backward()               # 反向传播，为什么不拿train_loss做反向传播？因为这里的train_loss只是一个值，没有backward()方法
            if accumulate and (cnt_train + 1) % a_step == 0:     # cnt += 1 见后面139行()，（16-1）次后梯度清零
                optimizer.step()    #权重更新
                optimizer.zero_grad()    #梯度清零
            if not accumulate:
                optimizer.step()    #不管怎么样，权重都会更新，只是前面梯度没有清零
            if use_dtp:
                good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = loss_object.correct_cnt_each(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                tmp_good = sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                tmp_total = sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                train_ocemotion_correct += good_ocemotion_nb
                train_ocnli_correct += good_ocnli_nb
                train_tnews_correct += good_tnews_nb
            else:
                tmp_good, tmp_total = loss_object.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            train_correct += tmp_good
            train_total += tmp_total
            p, g = loss_object.collect_pred_and_gold(ocnli_pred, ocnli_gold)
            train_ocnli_pred_list += p
            train_ocnli_gold_list += g
            p, g = loss_object.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
            train_ocemotion_pred_list += p
            train_ocemotion_gold_list += g
            p, g = loss_object.collect_pred_and_gold(tnews_pred, tnews_gold)
            train_tnews_pred_list += p
            train_tnews_gold_list += g
            cnt_train += 1
            #torch.cuda.empty_cache()
            if (cnt_train + 1) % 1000 == 0:
                print('[', cnt_train + 1, '- th batch : train acc is:', train_correct / train_total, '; train loss is:', train_loss / cnt_train, ']')
        if accumulate:
            optimizer.step()
        optimizer.zero_grad()
        train_ocnli_f1 = get_f1(train_ocnli_gold_list, train_ocnli_pred_list)
        train_ocemotion_f1 = get_f1(train_ocemotion_gold_list, train_ocemotion_pred_list)
        train_tnews_f1 = get_f1(train_tnews_gold_list, train_tnews_pred_list)
        train_avg_f1 = (train_ocnli_f1 + train_ocemotion_f1 + train_tnews_f1) / 3
        print(epoch, 'th epoch train average f1 is:', train_avg_f1)
        print(epoch, 'th epoch train ocnli is below:')
        print_result(train_ocnli_gold_list, train_ocnli_pred_list)
        print(epoch, 'th epoch train ocemotion is below:')
        print_result(train_ocemotion_gold_list, train_ocemotion_pred_list)
        print(epoch, 'th epoch train tnews is below:')
        print_result(train_tnews_gold_list, train_tnews_pred_list)
        
        train_data_generator.reset()
        
        my_net.eval()
        dev_loss = 0.0
        dev_total = 0
        dev_correct = 0
        dev_ocnli_correct = 0
        dev_ocemotion_correct = 0
        dev_tnews_correct = 0
        dev_ocnli_pred_list = []
        dev_ocnli_gold_list = []
        dev_ocemotion_pred_list = []
        dev_ocemotion_gold_list = []
        dev_tnews_pred_list = []
        dev_tnews_gold_list = []
        cnt_dev = 0
        with torch.no_grad():
            while True:
                raw_data = dev_data_generator.get_next_batch(batchSize)
                if raw_data == None:
                    break
                data = dict()
                data['input_ids'] = raw_data['input_ids']
                data['token_type_ids'] = raw_data['token_type_ids']
                data['attention_mask'] = raw_data['attention_mask']
                data['ocnli_ids'] = raw_data['ocnli_ids']
                data['ocemotion_ids'] = raw_data['ocemotion_ids']
                data['tnews_ids'] = raw_data['tnews_ids']
                tnews_gold = raw_data['tnews_gold']
                ocnli_gold = raw_data['ocnli_gold']
                ocemotion_gold = raw_data['ocemotion_gold']
                ocnli_pred, ocemotion_pred, tnews_pred = my_net(**data)
                if use_dtp:
                    tnews_kpi = 0.1 if len(dev_tnews_pred_list) == 0 else dev_tnews_correct / len(
                        dev_tnews_pred_list)
                    ocnli_kpi = 0.1 if len(dev_ocnli_pred_list) == 0 else dev_ocnli_correct / len(
                        dev_ocnli_pred_list)
                    ocemotion_kpi = 0.1 if len(dev_ocemotion_pred_list) == 0 else dev_ocemotion_correct / len(
                        dev_ocemotion_pred_list)
                    current_loss = loss_object.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold,
                                                           ocnli_gold,
                                                           ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
                else:
                    current_loss = loss_object.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_loss += current_loss.item()
                if use_dtp:
                    good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = loss_object.correct_cnt_each(
                        tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                    tmp_good += sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                    tmp_total += sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                    dev_ocemotion_correct += good_ocemotion_nb
                    dev_ocnli_correct += good_ocnli_nb
                    dev_tnews_correct += good_tnews_nb
                else:
                    tmp_good, tmp_total = loss_object.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_correct += tmp_good
                dev_total += tmp_total
                p, g = loss_object.collect_pred_and_gold(ocnli_pred, ocnli_gold)
                dev_ocnli_pred_list += p
                dev_ocnli_gold_list += g
                p, g = loss_object.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
                dev_ocemotion_pred_list += p
                dev_ocemotion_gold_list += g
                p, g = loss_object.collect_pred_and_gold(tnews_pred, tnews_gold)
                dev_tnews_pred_list += p
                dev_tnews_gold_list += g
                cnt_dev += 1
                #torch.cuda.empty_cache()
                #if (cnt_dev + 1) % 1000 == 0:
                #    print('[', cnt_dev + 1, '- th batch : dev acc is:', dev_correct / dev_total, '; dev loss is:', dev_loss / cnt_dev, ']')
            dev_ocnli_f1 = get_f1(dev_ocnli_gold_list, dev_ocnli_pred_list)
            dev_ocemotion_f1 = get_f1(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            dev_tnews_f1 = get_f1(dev_tnews_gold_list, dev_tnews_pred_list)
            dev_avg_f1 = (dev_ocnli_f1 + dev_ocemotion_f1 + dev_tnews_f1) / 3
            print(epoch, 'th epoch dev average f1 is:', dev_avg_f1)
            print(epoch, 'th epoch dev ocnli is below:')
            print_result(dev_ocnli_gold_list, dev_ocnli_pred_list)
            print(epoch, 'th epoch dev ocemotion is below:')
            print_result(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            print(epoch, 'th epoch dev tnews is below:')
            print_result(dev_tnews_gold_list, dev_tnews_pred_list)

            dev_data_generator.reset()
            
            if dev_avg_f1 > best_dev_f1:
                best_dev_f1 = dev_avg_f1
                best_epoch = epoch
                torch.save(my_net, file_path)
            print('best epoch is:', best_epoch, '; with best f1 is:', best_dev_f1)
                
if __name__ == '__main__':
    print('---------------------start training-----------------------')
    pretrained_model = './bert_pretrain_model'
    tokenizer_model = './bert_pretrain_model'
    train(batchSize=1, device='cuda:0', lr=0.0001, use_dtp=True, pretrained_model=pretrained_model, tokenizer_model=tokenizer_model, weighted_loss=True)