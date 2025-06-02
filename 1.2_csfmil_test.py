'''
test csf-mil
'''

import csv
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
import torch
from torch.utils.data import DataLoader

from dataloader import read_bag, m3MILDataset
from model import Attention_fea,Siamese_Net,Attention_fea_2,Classifier_head

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # path to feature
    feature_path_a = ''  # scale 1, eg.:5*
    feature_path_b = ''  # scale 2, eg.:10*
    feature_path_c = ''  # scale 3, eg.:20*

    # path to fold&label
    fold_path = 'label/xxx.csv'

    # test log
    result_text = open("./evaluation/result_slice.txt", "a")
    test_text = open("./evaluation/test_slice.txt", "a")
    nw = 32

    csvfile = open(fold_path,encoding='UTF-8-sig')
    set_csv = csv.reader(csvfile)
    set_list = []

    acc_l = []
    auc_l = []
    precision_l = []
    recall_l = []
    specificity_l = []
    f1_l = []
    tpr_l = []
    fpr_m = np.linspace(0, 1, 100)

    for i, rows in enumerate(set_csv):
        set_list.append(rows)

    for s in set_list:
        test_list = s

        net_a = Attention_fea().to(device)
        net_b = Attention_fea().to(device)
        net_c = Attention_fea().to(device)
        net_sn = Siamese_Net().to(device)
        net_cs = Attention_fea_2().to(device)
        net_d = Classifier_head().to(device)

        weights_path = './evaluation/weight_los_' + str(set_list.index(s)) + '.pth'

        model = torch.load(weights_path)
        net_a.load_state_dict(model['net_a'])
        net_b.load_state_dict(model['net_b'])
        net_c.load_state_dict(model['net_c'])
        net_sn.load_state_dict(model['net_sn'])
        net_cs.load_state_dict(model['net_cs'])
        net_d.load_state_dict(model['net_d'])

        test_patient_list_a, test_label_list_a = read_bag(test_list, feature_path_a)
        test_patient_list_b, test_label_list_b = read_bag(test_list, feature_path_b)
        test_patient_list_c, test_label_list_c = read_bag(test_list, feature_path_c)

        test_df = pd.DataFrame(
            {'bag_a': test_patient_list_a, 'bag_b': test_patient_list_b, 'bag_c': test_patient_list_c,'label': test_label_list_a})

        test_dataset = m3MILDataset(test_df)
        test_num = len(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle = True, num_workers=nw)
        print("Fold:{}, test_patient:{}".format(str(set_list.index(s)), test_num))
        print("Fold:{}, test_patient:{}".format(str(set_list.index(s)), test_num), file=result_text)

        result = {'patient': [], 'predict': [], 'label': [], 'bag_score': []}
        net_a.eval()
        net_b.eval()
        net_c.eval()
        net_sn.eval()
        net_cs.eval()
        net_d.eval()

        with torch.no_grad():

            for test_data in test_loader:
                test_bags_a, test_bags_b, test_bags_c, test_labels = test_data

                fea_a = net_a(test_bags_a.to(device))
                fea_b = net_b(test_bags_b.to(device))
                fea_c = net_c(test_bags_c.to(device))

                fea_cs_a = net_sn(fea_a)
                fea_cs_b = net_sn(fea_b)
                fea_cs_c = net_sn(fea_c)

                cs_bag = torch.cat([1 * fea_cs_a, 1 * fea_cs_b, 1 * fea_cs_c], 0).unsqueeze(0)
                shared_feature = net_cs(cs_bag)
                bag_feature = torch.cat([1 * fea_cs_a, 1 * fea_cs_b, 1 * fea_cs_c, 1 * shared_feature], 1)

                bag_prob, predict_t = net_d(bag_feature.to(device))

                result['patient'].append('patient')
                result['predict'].append(int(predict_t))
                result['label'].append(int(test_labels))
                result['bag_score'].append(bag_prob[0][0].cpu().numpy().tolist())

                print("patient:{}, label:{}, predict:{}, bag_prob:{}".
                      format('patient', int(test_labels), int(predict_t), bag_prob
                             ),
                      file=result_text
                      )

        # 计算评价指标
        fpr, tpr, thresholds = roc_curve(result['label'], result['bag_score'], pos_label=1)
        cm = confusion_matrix(result['label'], result['predict']) # cm.ravel(): tn,fp,fn,tp
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(result['label'], result['predict'])
        acc_l.append(acc)
        precision = precision_score(result['label'], result['predict'])
        precision_l.append(precision)
        recall = recall_score(result['label'], result['predict'])
        recall_l.append(recall)
        f1 = f1_score(result['label'], result['predict'])
        f1_l.append(f1)
        specificity = tn /(fp + tn)
        specificity_l.append(specificity)

        tpr_l.append(interp(fpr_m, fpr, tpr))
        tpr_l[-1][0] = 0.0
        rauc = auc(fpr, tpr)
        auc_l.append(rauc)

        print("Fold:{}, acc:{}, auc:{}, pre:{}, recall:{}, spec:{}, f1:{}".format(str(set_list.index(s)), acc, rauc, precision, recall, specificity, f1))
        print("Fold:{}, acc:{:.3f}, auc:{:.3f}, pre:{:.3f}, recall:{:.3f}, spec:{:.3f}, f1:{:.3f}".format(str(set_list.index(s)), acc, rauc, precision, recall, specificity, f1), file=test_text)

        plt.plot(fpr, tpr, label='AUC {}-fold = {:.3f}'.format(set_list.index(s),rauc))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.title('ROC curve_fold {:.3f}'.format(set_list.index(s)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    acc_m = mean(acc_l)
    precision_m = mean(precision_l)
    recall_m = mean(recall_l)
    specificity_m = mean(specificity_l)
    f1_m = mean(f1_l)
    tpr_m = mean(tpr_l, axis=0)
    tpr_m[-1] = 1.0

    auc_m = auc(fpr_m, tpr_m)

    print("Mean: acc:{}, auc:{}, pre:{}, recall:{}, spec:{}, f1:{}".format(acc_m, auc_m, precision_m, recall_m, specificity_m, f1_m))
    print("Mean: acc:{:.3f}, auc:{:.3f}, pre:{:.3f}, recall:{:.3f}, spec:{:.3f}, f1:{:.3f}".format(acc_m, auc_m, precision_m, recall_m, specificity_m, f1_m),
          file=test_text)

    plt.plot(fpr_m, tpr_m, label='AUC mean = {:.3f}'.format(auc_m))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('ROC curve_mean')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('./evaluation/ROC_mean.png')
    # plt.show()
