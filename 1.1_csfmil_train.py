'''
train csf-mil
'''

import csv
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataloader import label_separate, read_bag, m3MILDataset
from model import Attention_fea,Siamese_Net,Attention_fea_2,Classifier_head

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # path to feature
    feature_path_a = '' #scale 1, eg.:5*
    feature_path_b = '' #scale 2, eg.:10*
    feature_path_c = '' #scale 3, eg.:20*

    # path to fold&label
    fold_path = 'label/xxx.csv'

    # train log
    train_text = open("./evaluation/train.txt", "a")

    # parameters
    batch_size = 1
    epochs = 100
    patience = 20
    lr = 1e-3
    wd = 1e-5
    nw = 4
    rs = 2 # random seed for validation split

    print("using {} device.".format(device))
    print("using {} device.".format(device), file=train_text)

    # load csv
    csvfile = open(fold_path, encoding='UTF-8-sig')
    set_csv = csv.reader(csvfile)
    set_list = []
    for i, rows in enumerate(set_csv):  # eg.: i: set number, rows: ["['3', 0]", "['30', 1]"]
        set_list.append(rows)

    for s in set_list:

        test_list = s
        training_list = [i for i in list(chain.from_iterable(set_list)) if i not in test_list]  # all - test

        print("Fold {}.".format(str(set_list.index(s))), file=train_text)
        list_0, list_1 = label_separate(training_list)

        train_list_0, valid_list_0 = train_test_split(list_0, test_size=0.1, random_state=rs)
        train_list_1, valid_list_1 = train_test_split(list_1, test_size=0.1, random_state=rs)

        train_list = train_list_0 + train_list_1
        valid_list = valid_list_0 + valid_list_1

        random.shuffle(train_list)
        random.shuffle(valid_list)
        weight_0 = len(train_list_1) / len(train_list)
        weight_1 = len(train_list_0) / len(train_list)

        train_patient_list_a, train_label_list_a = read_bag(train_list, feature_path_a)
        valid_patient_list_a, valid_label_list_a = read_bag(valid_list, feature_path_a)
        train_patient_list_b, train_label_list_b = read_bag(train_list, feature_path_b)
        valid_patient_list_b, valid_label_list_b = read_bag(valid_list, feature_path_b)
        train_patient_list_c, train_label_list_c = read_bag(train_list, feature_path_c)
        valid_patient_list_c, valid_label_list_c = read_bag(valid_list, feature_path_c)

        train_df = pd.DataFrame(
            {'bag_a': train_patient_list_a, 'bag_b': train_patient_list_b, 'bag_c': train_patient_list_c, 'label': train_label_list_a})
        valid_df = pd.DataFrame(
            {'bag_a': valid_patient_list_a, 'bag_b': valid_patient_list_b, 'bag_c': valid_patient_list_c, 'label': valid_label_list_a})

        train_dataset = m3MILDataset(train_df)
        valid_dataset = m3MILDataset(valid_df)

        train_num = len(train_dataset)
        val_num = len(valid_dataset)

        print('Using {} dataloader workers every process'.format(nw))
        print('Using {} dataloader workers every process'.format(nw), file=train_text)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        print("using {} bags for training, {} bags for validation.".format(train_num, val_num))
        print("using {} bags for training, {} bags for validation.".format(train_num, val_num), file=train_text)

        net_a = Attention_fea().to(device) # 5*
        net_b = Attention_fea().to(device) # 10*
        net_c = Attention_fea().to(device) # 20*
        net_sn = Siamese_Net().to(device)
        net_cs = Attention_fea_2().to(device) # cross-scale
        net_d = Classifier_head().to(device)

        loss_function = nn.BCELoss()

        optimizer = optim.Adam([
            {"params": net_a.parameters()},
            {"params": net_b.parameters()},
            {"params": net_c.parameters()},
            {"params": net_sn.parameters()},
            {"params": net_cs.parameters()},
            {"params": net_d.parameters()}
        ], lr=lr, weight_decay=wd)

        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5,min_lr=1e-8)

        # train & val
        best_acc = 0.0
        best_loss = 100
        best_auc = 0.0
        train_steps = len(train_loader)
        history = {'train_loss': [], 'val_loss': [], 'train_acc':[], 'val_acc': []}
        for epoch in range(epochs):

            # train
            print('lr = ', optimizer.state_dict()['param_groups'][0]['lr'])

            net_a.train()
            net_b.train()
            net_c.train()
            net_sn.train()
            net_cs.train()
            net_d.train()

            acc_t = 0.0
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            trn_0 = 0
            trn_1 = 0
            val_0 = 0
            val_1 = 0
            trn_rt_0 = 0
            trn_rt_1 = 0
            val_rt_0 = 0
            val_rt_1 = 0

            for step, data in enumerate(train_bar):
                bags_a, bags_b, bags_c, labels = data

                bags_a, bags_b, bags_c, labels = Variable(bags_a), Variable(bags_b), Variable(bags_c), Variable(labels)

                if labels == 0:
                    trn_0 += 1
                if labels == 1:
                    trn_1 += 1

                optimizer.zero_grad()

                fea_a = net_a(bags_a.to(device))
                fea_b = net_b(bags_b.to(device))
                fea_c = net_c(bags_c.to(device))
                fea_cs_a = net_sn(fea_a)
                fea_cs_b = net_sn(fea_b)
                fea_cs_c = net_sn(fea_c)

                cs_bag = torch.cat([1 * fea_cs_a, 1 * fea_cs_b, 1 * fea_cs_c], 0).unsqueeze(0) # 1, 3, 512
                shared_feature = net_cs(cs_bag)
                bag_feature = torch.cat([1 * fea_cs_a, 1 * fea_cs_b, 1 * fea_cs_c, 1 * shared_feature], 1)  # concat
                bag_prob, predict_t = net_d(bag_feature.to(device))

                # weighted loss
                weight = torch.zeros_like(labels).float().to(device)
                weight = torch.fill_(weight, weight_0)
                weight[labels == 1] = weight_1
                loss = nn.BCELoss(weight=weight)(bag_prob[0], labels.to(device).to(torch.float32))

                running_loss += loss.item()
                acc_t += torch.eq(predict_t, labels.to(device)).sum().item()
                loss.backward()
                optimizer.step()

                if predict_t == labels.to(device):
                    if labels == 1:
                        trn_rt_1 += 1
                    if labels == 0:
                        trn_rt_0 += 1
            print('train right prediction 0:', trn_rt_0, '/', trn_0)
            print('train right prediction 1:', trn_rt_1, '/', trn_1)
            print('train right prediction 0:', trn_rt_0, '/', trn_0, file=train_text)
            print('train right prediction 1:', trn_rt_1, '/', trn_1, file=train_text)
            train_loss = running_loss / len(train_loader)
            train_accurate = acc_t / len(train_loader.dataset)

            # validate
            net_a.eval()
            net_b.eval()
            net_c.eval()
            net_sn.eval()
            net_cs.eval()
            net_d.eval()
            acc_val = 0.0
            running_valid_loss = 0
            with torch.no_grad():
                val_bar = tqdm(valid_loader)
                result = {'label': [], 'bag_score': []}
                for val_data in val_bar:
                    val_bags_a, val_bags_b, val_bags_c, val_labels = val_data

                    if val_labels == 0:
                        val_0 += 1
                    if val_labels == 1:
                        val_1 += 1

                    val_fea_a = net_a(val_bags_a.to(device))
                    val_fea_b = net_b(val_bags_b.to(device))
                    val_fea_c = net_c(val_bags_c.to(device))

                    val_fea_cs_a = net_sn(val_fea_a)
                    val_fea_cs_b = net_sn(val_fea_b)
                    val_fea_cs_c = net_sn(val_fea_c)

                    val_cs_bag = torch.cat([1 * val_fea_cs_a, 1 * val_fea_cs_b, 1 * val_fea_cs_c], 0).unsqueeze(0)  # 1, 3 512
                    val_shared_feature = net_cs(val_cs_bag)
                    val_bag_feature = torch.cat([1 * val_fea_cs_a, 1 * val_fea_cs_b, 1 * val_fea_cs_c, 1 * val_shared_feature],
                                            1)  # concat
                    val_prob, predict_v = net_d(val_bag_feature.to(device))

                    weight = torch.zeros_like(val_labels).float().to(device)
                    weight = torch.fill_(weight, weight_0)
                    weight[val_labels == 1] = weight_1
                    loss_v = nn.BCELoss(weight=weight)(val_prob[0], val_labels.to(device).to(torch.float32))

                    running_valid_loss += loss_v.item()
                    acc_val += torch.eq(predict_v, val_labels.to(device)).sum().item()

                    if predict_v == val_labels.to(device):
                        if val_labels == 1:
                            val_rt_1 += 1
                        if val_labels == 0:
                            val_rt_0 += 1
                    result['label'].append(int(val_labels))
                    result['bag_score'].append(val_prob[0][0].cpu().numpy().tolist())  # tensor to numpy to float

            fpr, tpr, thresholds = roc_curve(result['label'], result['bag_score'], pos_label=1)  # pos_label：指定正标签
            auc_val = auc(fpr, tpr)  # auc

            print('val auc:',auc_val)

            print('val right prediction 0:', val_rt_0, '/', val_0)
            print('val right prediction 1:', val_rt_1, '/', val_1)
            print('val right prediction 0:', val_rt_0, '/', val_0, file=train_text)
            print('val right prediction 1:', val_rt_1, '/', val_1, file=train_text)
            val_loss = running_valid_loss / len(valid_loader)
            val_accurate = acc_val / len(valid_loader.dataset)
            lr_scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_accurate)
            history['val_acc'].append(val_accurate)

            print('[epoch %d] train_loss: %.3f  val_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_loss, train_accurate, val_accurate))
            print('[epoch %d] train_loss: %.3f  val_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_loss, train_accurate, val_accurate), file=train_text)

            state = {'net_a': net_a.state_dict(),
                     'net_b': net_b.state_dict(),
                     'net_c': net_c.state_dict(),
                     'net_sn': net_sn.state_dict(),
                     'net_cs': net_cs.state_dict(),
                     'net_d': net_d.state_dict()}

            # early stopping
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(state, './evaluation/weight_acc_' + str(set_list.index(s)) + '.pth')
            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(state, './evaluation/weight_auc_' + str(set_list.index(s)) + '.pth')
            if val_loss < best_loss:
                best_loss = val_loss
                es = 0
                torch.save(state, './evaluation/weight_los_' + str(set_list.index(s)) + '.pth')
            else:
                es += 1
                print("Counter {} of {}".format(es, patience))
                if es > int(patience):
                    print("Early stopping with best_acc: {}, and val_acc for this epoch:{}".format(best_acc, val_accurate))
                    break

        plt.figure(figsize=(7, 7))
        plt.plot(history['train_loss'], label='Training loss')
        plt.plot(history['val_loss'], label='Validation loss')
        plt.legend()
        plt.savefig('./evaluation/loss_' + str(set_list.index(s)) + '.png')
        # plt.show()

        plt.figure(figsize=(7, 7))
        plt.plot(history['train_acc'], label='train_acc')
        plt.plot(history['val_acc'], label='valid_acc')
        plt.legend()
        plt.savefig('./evaluation/val_' + str(set_list.index(s)) + '.png')
        # plt.show()

    print('Finished Training')
