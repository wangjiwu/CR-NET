import PersonalityDataset
from torch.utils.data import TensorDataset
#import librosa
from PersonalityDataset import get_data_list_and_big_five,get_one_hot
import new_resnet_6d
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from new_resnet_6d import ResNet_audio, ResNet_old_pretrain
import time
from BellLoss import BellingLoss
from config import useHead, without_closs,modality_type, resnet_pretrained,full_train,div_arr, fix_step1_weight,train_stage,device_num,random_seclect
from valid_resnet import test_audio_resnet ,test_visual,extra_tree_regress_eval
import pickle
import sys, os,random
import argparse
import  csv
from new_resnet_6d import get_parameter_number

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler

import logging

# 1.显示创建
logging.basicConfig(filename='step2_logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 2.定义logger,设定setLevel，FileHandler，setFormatter
logger = logging.getLogger(__name__)  # 定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
logger.setLevel(level=logging.INFO)  # 定义过滤级别
filehandler = logging.FileHandler("step2_log" + "_modality_type_" + str(modality_type) +  ".txt")  # Handler用于将日志记录发送至合适的目的地，如文件、终端等
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler()  # 日志信息显示在终端terminal
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)


if modality_type == 2:
    BAST_RESULT = 0.895
else:
    BAST_RESULT = 0.910


epoch_max_number = 50

train_batch_size = 30

#learning_rate = 0.00002

learning_rate = 0.00002

momentum = 0.9
num_classes = 4
weight_decay = 0.005
torch.cuda.set_device(device_num)


logger.info({"fix_step1_weight": fix_step1_weight,
             "random_seclect":random_seclect,
             "BAST_RESULT":BAST_RESULT,
             "epoch_max_number":epoch_max_number,
             "train_batch_size":train_batch_size,
             "learning_rate":learning_rate,
             "modality_type":modality_type,
             "momentum":momentum,
             "num_classes":num_classes,
             "weight_decay":weight_decay,
             "device_num":device_num,
             "resnet_pretrained":resnet_pretrained,
             "full_train":full_train,
             "div_arr":div_arr,
             })



train_plot_arr_allloss = []
valid_plot_arr_allloss = []
test_plot_arr_allloss = []



if modality_type == 0:

    model = new_resnet_6d.resnet34_old(pretrained=False
                                       ,num_output=4)

    model_name = "./models/class_best_shit_14.3079_modality_type_0.pt"
    model.load_state_dict(torch.load(model_name), False)
    logger.info(model_name)
elif modality_type == 1:

    model = new_resnet_6d.resnet34_old(pretrained=False
                                       ,num_output=4)

    model_name = "./models/class_best_shit_13.8237_modality_type_1.pt"
    model.load_state_dict(torch.load(model_name), False)
    logger.info(model_name)

else :

    model = new_resnet_6d.resnet34_audio(pretrained=False,num_output=4)

    model_name = "./models/class_best_shit_15.1236_modality_type_2.pt"
    model.load_state_dict(torch.load(model_name), False)
    logger.info(model_name)




model = model.cuda()


#  defining the criterion & optimizer
loss = torch.nn.CrossEntropyLoss()
L2loss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()
bellloss = BellingLoss()

featconv_e_params = list(map(id, model.featconv_e.parameters()))
featconv_n_params = list(map(id, model.featconv_n.parameters()))
featconv_a_params = list(map(id, model.featconv_a.parameters()))
featconv_c_params = list(map(id, model.featconv_c.parameters()))
featconv_o_params = list(map(id, model.featconv_o.parameters()))
featconv_i_params = list(map(id, model.featconv_i.parameters()))

rfc_e_params = list(map(id, model.rfc_e.parameters()))
rfc_n_params = list(map(id, model.rfc_n.parameters()))
rfc_a_params = list(map(id, model.rfc_a.parameters()))
rfc_c_params = list(map(id, model.rfc_c.parameters()))
rfc_o_params = list(map(id, model.rfc_o.parameters()))
rfc_i_params = list(map(id, model.rfc_i.parameters()))

base_params = filter(lambda p: id(p) not in featconv_e_params+featconv_n_params+featconv_a_params+featconv_c_params+featconv_o_params+featconv_i_params
                     +rfc_e_params+rfc_n_params+rfc_a_params+rfc_c_params+rfc_o_params+rfc_i_params,
                     model.parameters())

optimizer = torch.optim.Adam([{'params': base_params},
                              {'params': model.featconv_e.parameters(), 'lr': learning_rate * 100},
                              {'params': model.featconv_n.parameters(), 'lr': learning_rate * 100},
                              {'params': model.featconv_a.parameters(), 'lr': learning_rate * 100},
                              {'params': model.featconv_c.parameters(), 'lr': learning_rate * 100},
                              {'params': model.featconv_o.parameters(), 'lr': learning_rate * 100},
                              {'params': model.featconv_i.parameters(), 'lr': learning_rate * 100},
                              {'params': model.rfc_e.parameters(), 'lr': learning_rate * 100},
                              {'params': model.rfc_n.parameters(), 'lr': learning_rate * 100},
                              {'params': model.rfc_a.parameters(), 'lr': learning_rate * 100},
                              {'params': model.rfc_c.parameters(), 'lr': learning_rate * 100},
                              {'params': model.rfc_o.parameters(), 'lr': learning_rate * 100},
                             {'params': model.rfc_i.parameters(), 'lr': learning_rate * 100}], lr=learning_rate, betas=[0.5, 0.999], weight_decay=weight_decay)

if fix_step1_weight:
    base_params = filter(lambda p: p.requires_grad, model.parameters())
    torch.optim.Adam(base_params, lr=learning_rate *100,
                     betas=[0.5, 0.999], weight_decay=weight_decay)

logger.info(get_parameter_number(model))


#  data preparation


data_list = get_data_list_and_big_five('./dataset/train', 'train')
random.shuffle (data_list )
data_list_train = data_list[0: 5400]
data_list_valid = data_list[5400: ]
test_data_list = get_data_list_and_big_five('./dataset/test', 'test')
#data_list = data_list[0:200]
#test_data_list = test_data_list[0:200]

#random.shuffle(data_list)

import torchvision

def change_lr(opt, learning_rate_tmp):
    learning_rate_tmp *= 0.1
    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate_tmp
    return learning_rate_tmp


def train(data_loader, epoch):


    total_closs = 0.0
    total_rloss = 0.0
    total_running_loss = 0.0
    total_L1loss = 0.0
    total_L2loss = 0.0
    total_Beloss = 0.0
    start_time = time.time()


    for batch, (arr, choice_frame, name, cls_labels, reg_labels) in enumerate(data_loader):




        model.train()


        el, nl, al, cl, ol, il = reg_labels
        ecl, ncl, acl, ccl, ocl, icl = cls_labels
        x_cls, x_reg, _ = model(arr.cuda())

        #预测的分类值
        ec = x_cls[0]
        nc = x_cls[1]
        ac = x_cls[2]
        cc = x_cls[3]
        oc = x_cls[4]
        ic = x_cls[5]


        # 计算分类损失
        lc_e = loss(ec, ecl.cuda())
        lc_n = loss(nc, ncl.cuda())
        lc_a = loss(ac, acl.cuda())
        lc_c = loss(cc, ccl.cuda())
        lc_o = loss(oc, ocl.cuda())
        lc_i = loss(ic, icl.cuda())
        # 真实值
        e = x_reg[0]
        n = x_reg[1]
        a = x_reg[2]
        c = x_reg[3]
        o = x_reg[4]
        i = x_reg[5]


        # 计算回归损失
        loss_e = L2loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                L1loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                 bellloss(torch.mul(e, 100), el.float().view_as(e).cuda())


        loss_n = L2loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                L1loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                bellloss(torch.mul(n, 100), nl.float().view_as(n).cuda())


        loss_a = L2loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                L1loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                bellloss(torch.mul(a, 100), al.float().view_as(a).cuda())


        loss_c = L2loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                L1loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
               bellloss(torch.mul(c, 100), cl.float().view_as(c).cuda())

        loss_o = L2loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                L1loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                bellloss(torch.mul(o, 100), ol.float().view_as(o).cuda())

        loss_i = L2loss(torch.mul(i, 100), il.float().view_as(i).cuda()) + \
                L1loss(torch.mul(i, 100), il.float().view_as(i).cuda()) + \
                bellloss(torch.mul(i, 100), il.float().view_as(i).cuda())

        # 分类损失
        total_L1loss = total_L1loss + \
                       (L1loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                       L1loss(torch.mul(n, 100), nl.float().view_as(n).cuda())+ \
                       L1loss(torch.mul(a, 100), al.float().view_as(a).cuda())+ \
                       L1loss(torch.mul(c, 100), cl.float().view_as(c).cuda())+ \
                       L1loss(torch.mul(o, 100), ol.float().view_as(o).cuda())+ \
                       L1loss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size
        total_L2loss = total_L1loss + \
                       (L2loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                        L2loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                        L2loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                        L2loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                        L2loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                        L2loss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size
        total_Beloss = total_L1loss + \
                       (bellloss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                        bellloss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                        bellloss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                        bellloss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                        bellloss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                        bellloss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size

        #.zero_grad()


        cls_loss = lc_e + lc_n + lc_a + lc_c + lc_o + lc_i



        # 回归损失
        reg_loss = loss_e + loss_n + loss_a + loss_c + loss_o + loss_i
        # 总损失

        if without_closs:
            running_loss =  reg_loss
        else:
            running_loss = cls_loss * epoch_max_number * 4 / (epoch+1) + reg_loss
            #running_loss = cls_loss * 200  + reg_loss

        # running_loss = cls_loss
        # optimizer_class.zero_grad()
        #
        # cls_loss.backward(retain_graph=True)
        # optimizer_class.step()

        optimizer.zero_grad()

        running_loss.backward()

        optimizer.step()


        total_closs = total_closs + cls_loss.item()/train_batch_size
        total_rloss = total_rloss + reg_loss.item() / train_batch_size
        total_running_loss = total_running_loss + running_loss.item() / train_batch_size




    train_plot_arr_allloss.append([total_rloss, total_closs, total_rloss, total_L1loss, total_L2loss, total_Beloss])



    if (epoch + 1)%10  == 0:

        np.savetxt('train_plot_arr_allloss' + str(useHead) + '.csv',train_plot_arr_allloss,
                   delimiter=',', fmt='%.5f')

    logger.info("epoch: " + str(epoch + 1))
    logger.info ( "=============training=====================")
    logger.info('total allLoss:  %4f  total rloss:  %4f   total closs:  %4f ' % (total_running_loss, total_rloss, total_closs,))


def valid(data_loader, epoch):


    total_closs = 0.0
    total_rloss = 0.0
    total_running_loss = 0.0
    total_L1loss = 0.0
    total_L2loss = 0.0
    total_Beloss = 0.0
    total_valid_acc = 0.0
    start_time = time.time()

    model.eval()
    with torch.no_grad():

        for batch, (arr, choice_frame, name, cls_labels, reg_labels) in enumerate(data_loader):

            el, nl, al, cl, ol, il = reg_labels
            ecl, ncl, acl, ccl, ocl, icl = cls_labels
            x_cls, x_reg, _ = model(arr.cuda())
            #预测的分类值
            ec = x_cls[0]
            nc = x_cls[1]
            ac = x_cls[2]
            cc = x_cls[3]
            oc = x_cls[4]
            ic = x_cls[5]



            # 计算分类损失
            lc_e = loss(ec, ecl.cuda())
            lc_n = loss(nc, ncl.cuda())
            lc_a = loss(ac, acl.cuda())
            lc_c = loss(cc, ccl.cuda())
            lc_o = loss(oc, ocl.cuda())
            lc_i = loss(ic, icl.cuda())
            # 真实值
            e = x_reg[0]
            n = x_reg[1]
            a = x_reg[2]
            c = x_reg[3]
            o = x_reg[4]
            i = x_reg[5]


            # 计算回归损失
            loss_e = L2loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                    L1loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                     bellloss(torch.mul(e, 100), el.float().view_as(e).cuda())


            loss_n = L2loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                    L1loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                    bellloss(torch.mul(n, 100), nl.float().view_as(n).cuda())


            loss_a = L2loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                    L1loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                    bellloss(torch.mul(a, 100), al.float().view_as(a).cuda())


            loss_c = L2loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                    L1loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                   bellloss(torch.mul(c, 100), cl.float().view_as(c).cuda())

            loss_o = L2loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                    L1loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                    bellloss(torch.mul(o, 100), ol.float().view_as(o).cuda())

            loss_i = L2loss(torch.mul(i, 100), il.float().view_as(i).cuda()) + \
                    L1loss(torch.mul(i, 100), il.float().view_as(i).cuda()) + \
                    bellloss(torch.mul(i, 100), il.float().view_as(i).cuda())

            # 分类损失
            total_L1loss = total_L1loss + \
                           (L1loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                           L1loss(torch.mul(n, 100), nl.float().view_as(n).cuda())+ \
                           L1loss(torch.mul(a, 100), al.float().view_as(a).cuda())+ \
                           L1loss(torch.mul(c, 100), cl.float().view_as(c).cuda())+ \
                           L1loss(torch.mul(o, 100), ol.float().view_as(o).cuda())+ \
                           L1loss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size
            total_L2loss = total_L1loss + \
                           (L2loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                            L2loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                            L2loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                            L2loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                            L2loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                            L2loss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size
            total_Beloss = total_L1loss + \
                           (bellloss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                            bellloss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                            bellloss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                            bellloss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                            bellloss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                            bellloss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size



            cls_loss = lc_e + lc_n + lc_a + lc_c + lc_o + lc_i

            # 回归损失
            reg_loss = loss_e + loss_n + loss_a + loss_c + loss_o + loss_i
            # 总损失

            if without_closs:
                running_loss =  reg_loss
            else:
                running_loss = cls_loss * epoch_max_number / (epoch+1) + reg_loss
                #running_loss = cls_loss * 200  + reg_loss

            # running_loss = cls_loss

            total_closs = total_closs + cls_loss.item()/train_batch_size
            total_rloss = total_rloss + reg_loss.item() / train_batch_size
            total_running_loss = total_running_loss + running_loss.item() / train_batch_size

            # logger.info('Eval [current epoch: %2d total: %2d iter/all %4d/%4d]  Loss: %.4f \t time: %.3f @ %s' % (epoch, epoch_max_number, batch,
            #                                                                len(data_loader.dataset) / train_batch_size,
            #                                                                 running_loss.item(), time.time() - start_time,
            #                                                                     time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
            # logger.info('batch total loss:  %4f  %4f  %4f ' % (cls_loss.item() ,reg_loss.item(), running_loss.item()))
            # logger.info('%')

        valid_plot_arr_allloss.append([total_running_loss, total_closs, total_rloss, total_L1loss, total_L2loss, total_Beloss])

    np.savetxt('valid_plot_arr_allloss' + str(useHead) + '.csv', valid_plot_arr_allloss,
               delimiter=',', fmt='%.5f')

    logger.info ("=============validition=====================")
    logger.info('total allLoss:  %4f  total rloss:  %4f   total closs:  %4f ' % (total_running_loss, total_rloss, total_closs,))

    logger.info('extr. prediction data vs label')
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    logger.info(torch.mul(n, 100).squeeze(1).cpu().data.numpy())
    logger.info(nl.float().data.numpy())

    logger.info("\n\n")

def test(data_loader, train_loader, epoch):



    acc = [0., 0., 0., 0., 0., 0]
    feature_vector = []
    label_vector = []
    pre_label_vector = []

    model.eval()
    with torch.no_grad():

        for batch, (arr, choice_frame, name, cls_labels, labels) in enumerate(data_loader):
            ecl, ncl, acl, ccl, ocl, icl = cls_labels
            el, nl, al, cl, ol, il = labels
            # e = model(arr.cuda())
            x_cls, x_reg, x_regress_result = model(arr.cuda())

            ec = x_cls[0]
            nc = x_cls[1]
            ac = x_cls[2]
            cc = x_cls[3]
            oc = x_cls[4]
            ic = x_cls[5]

            # tmp = torch.stack([ec,nc,ac,cc,oc,ic])
            #
            # x_regress_result = torch.bmm(x_regress_result, tmp)

            for i in range(x_regress_result.shape[0]):
                feature_vector.append(x_regress_result[i].cpu().numpy())

                label_vector.append([el[i].item() / 100, nl[i].item() / 100, al[i].item() / 100, cl[i].item() / 100,
                                     ol[i].item() / 100, il[i].item() / 100])
                pre_label_vector.append(
                    [x_reg[0][i].cpu().numpy(), x_reg[1][i].cpu().numpy(), x_reg[2][i].cpu().numpy(),
                     x_reg[3][i].cpu().numpy(), x_reg[4][i].cpu().numpy(), x_reg[5][i].cpu().numpy()])


            # logger.info('extr. cls. prediction data vs label ', p_e.cpu().data.view(1, -1), ecl.cpu().data)
            #
            # logger.info('extr. prediction data vs label')
            # np.set_logger.infooptions(formatter={'float': '{: 0.1f}'.format})
            # logger.info(torch.mul(x_reg[1], 100).squeeze(1).cpu().data.numpy())
            # logger.info(nl.float().data.numpy())
            # logger.info("\n\n")
            #
            # logger.info('extr. reg. prediction data vs label ', x_reg[0].cpu().data.view(1, -1),
            #       torch.mul(el, 0.01).cpu().data)

            for i in range(0, 6):
                acc[i] += torch.sum(torch.div(
                    torch.abs((x_reg[i].cpu().data - torch.mul(labels[i], 0.01).float().view_as(x_reg[i]))),
                    1
                )).item()

    logger.info("=============testing=====================")
    result = []
    label_vector = np.array(label_vector)
    pre_label_vector = np.array(pre_label_vector)
    for i in range(6):
        x = label_vector[:, i]
        y = pre_label_vector[:, i]
        error = []
        for j in range(len(x)):
            error.append(abs(x[j] - y[j]))

        # logger.info(np.mean(error))
        result.append(np.mean(error))

    logger.info(result)
    logger.info(1 - np.mean(result[0:5]))

    train_feature_vector = []
    train_label_vector = []

    # 用训练好的模型测试  测试集

    with torch.no_grad():

        for batch, (arr, choice_frame, name, cls_labels, labels) in enumerate(train_loader):

            el, nl, al, cl, ol, il = labels

            x_cls, x_reg, x_regress_result = model(arr.cuda())


            for i in range(x_regress_result.shape[0]):
                train_feature_vector.append(x_regress_result[i].cpu().numpy())

                train_label_vector.append([el[i].item() / 100, nl[i].item() / 100, al[i].item() / 100, cl[i].item() / 100,
                                     ol[i].item() / 100, il[i].item() / 100])

    result_list = []

    train_feature_vector = np.array(train_feature_vector)
    feature_vector = np.array(feature_vector)
    train_label_vector = np.array(train_label_vector)
    label_vector = np.array(label_vector)
    for i in range(6):

        ss_x = StandardScaler()

        x_train = ss_x.fit_transform(train_feature_vector[:, :, i])
        x_test = ss_x.transform(feature_vector[:, :, i])

        y_train = train_label_vector[:, i]
        y_test = label_vector[:, i]

        etr = ExtraTreesRegressor(n_estimators=200, max_features=512)
        etr.fit(x_train, y_train)

        etr_y_predict = etr.predict(x_test)

        result = 0

        for i in range(len(etr_y_predict)):
            result = result + abs(etr_y_predict[i] - y_test[i])

        result = result / len(etr_y_predict)

        #logger.info("extra tree eval: " + str(1 - result))

        result_list.append(1 - result)

    total_L1loss = np.mean(result_list[0:5])
    logger.info(result_list)

    logger.info(total_L1loss)


    global BAST_RESULT

    if BAST_RESULT < total_L1loss:
        #BAST_RESULT = total_L1loss
        model_name = './models/best_shit_' + str(total_L1loss)[0:7]  + "_modality_type_" + str(modality_type) + '.model'

        logger.info("save the shit:  " + str(total_L1loss))
        torch.save(model, model_name)


    logger.info("\n\n")


def train_image():


    logger.info("total valid simple : " + str(len(data_list)))



    if full_train:

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list, useHead=useHead,
                                                              modality_type=modality_type)
    else:

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list_train, useHead=useHead,
                                                              modality_type=modality_type)
        valid_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list_valid , useHead=useHead,
                                                              modality_type=modality_type)

        valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)


    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         test_data_list, useHead=useHead,
                                                          modality_type=modality_type)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)



    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


    logger.info("train simple:  " + str(len(train_dataset.data_list)))

    logger.info("test simple:  " + str(len(test_dataset.data_list)))

    for epoch in range(epoch_max_number):

        global learning_rate

        if modality_type == 0:
            model_name = './models/567_pre_useSence_%d.model' % (epoch + 1)
        elif modality_type == 1 :
            model_name = './models/567_pre_useHead_%d.model' % (epoch + 1)
        else:
            model_name = './models/567_pre_useAudio_%d.model' % (epoch + 1)


        train(data_loader=train_dataloader, epoch=epoch)
        if not full_train:
            valid(data_loader=valid_dataloader, epoch=epoch)

        test(data_loader=test_dataloader, train_loader=train_dataloader, epoch=epoch)



        if (epoch + 1) % 10 == 0:

            learning_rate = change_lr(optimizer,
                                      learning_rate)  # here the reference of learning_rate is also globally changed
            logger.info("======================" + str(learning_rate) + "========================")

        if (epoch + 1) % 10 == 0 and epoch >= 20:
            torch.save(model, model_name)

    logger.info(model_name)

    logger.info_loss()

def train_text():
    typ = "train"
    with open('./dataset/%s/annotation_%s.pkl' % (typ, typ), 'rb') as fo:  # 读取pkl文件数据
        labels_train = pickle.load(fo, encoding='bytes')

    train_feature_vector = []
    train_label_vector = []
    test_feature_vector = []
    test_label_vector = []

    with open('feature_extraction_result/train_textfeature.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            #logger.info(row)
            filename = row[0]
            tmp = row[1:]

            feature_vec = []
            for item in tmp:
                feature_vec.append(float(item))

            train_feature_vector.append(feature_vec)
            train_label_vector.append([labels_train[b'openness'][str.encode(filename)] ,
                                    labels_train[b'agreeableness'][str.encode(filename)] ,
                                    labels_train[b'conscientiousness'][str.encode(filename)] ,
                                    labels_train[b'extraversion'][str.encode(filename)] ,
                                    labels_train[b'neuroticism'][str.encode(filename)]
                                    ])

    typ = "test"
    with open('./dataset/%s/annotation_%s.pkl' % (typ, typ), 'rb') as fo:  # 读取pkl文件数据
        labels_train = pickle.load(fo, encoding='bytes')
    with open('feature_extraction_result/test_textfeature.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            #logger.info(row)
            filename = row[0]
            tmp = row[1:]

            feature_vec = []
            for item in tmp:
                feature_vec.append(float(item))

            test_feature_vector.append(feature_vec)
            test_label_vector.append([labels_train[b'openness'][str.encode(filename)] ,
                                    labels_train[b'agreeableness'][str.encode(filename)] ,
                                    labels_train[b'conscientiousness'][str.encode(filename)] ,
                                    labels_train[b'extraversion'][str.encode(filename)] ,
                                    labels_train[b'neuroticism'][str.encode(filename)]
                                    ])


    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

    from sklearn.preprocessing import StandardScaler

    logger.info("train_feature_arr:       " + str(len(train_feature_vector)))
    logger.info("train_feature_score:     " + str(len(train_label_vector)))
    logger.info("test_feature_arr:        " + str(len(test_feature_vector)))
    logger.info("test_feature_score:      " + str(len(test_label_vector)))

    result_list = []

    train_label_vector = np.array(train_label_vector)
    test_label_vector = np.array(test_label_vector)

    for i in range(5):

        ss_x = StandardScaler()

        x_train = ss_x.fit_transform(train_feature_vector)
        x_test = ss_x.transform(test_feature_vector)

        y_train = train_label_vector[:, i]
        y_test = test_label_vector[:, i]

        etr = ExtraTreesRegressor(n_estimators=200,max_features=len(x_train[0]))
        etr.fit(x_train, y_train)

        etr_y_predict = etr.predict(x_test)

        result = 0

        for i in range(len(etr_y_predict)):
            result = result + abs(etr_y_predict[i] - y_test[i])

        result = result / len(etr_y_predict)

        logger.info("extra tree eval: " + str(1 - result))

        result_list.append(1 - result)
    logger.info(result_list)
    logger.info(np.mean(result_list))

def train_audio():

    global learning_rate
    useIS_OS = False
    if useIS_OS :
        import scipy.io as scio
        path = './feature_extraction_result/OS_IS13.mat'

        data = scio.loadmat(path)

        # 读取数据 和训练集视频对应 6000个
        # logger.info (data["OS_IS13"][0][0][0])

        dict_audio_data = {}

        for item in range(8000):
            file_name = data["OS_IS13"][0][0][0][item][0][0]
            # logger.info(file_name)
            # logger.info(data["OS_IS13"][0][0][1][item])
            dict_audio_data[file_name] = data["OS_IS13"][0][0][1][item]

        train_feature = []
        train_label = []
        for item in data_list:
            train_feature.append(dict_audio_data[item[0]])
            train_label.append(item[1:])
        train_feature = torch.Tensor(train_feature)
        train_label = torch.Tensor(train_label)
        train_ids = TensorDataset(train_feature, train_label)
        train_dataloader = DataLoader(train_ids, batch_size=train_batch_size, shuffle=True)

        for epoch in range(epoch_max_number):

            global learning_rate
            model_name = './models/audio_%d.model' % (epoch + 1)

            train_audio_OS_IS(data_loader=train_dataloader, epoch=epoch)
            model.eval()
            # valid(data_loader=valid_dataloader, epoch=epoch)
            # model.train()

            if (epoch + 1) % 10 == 0:
                learning_rate = change_lr(optimizer,
                                          learning_rate)  # here the reference of learning_rate is also globally changed
            if (epoch + 1) % 5 == 0:
                torch.save(model, model_name)
    else:
        #data_list = data_list[0:200]
        logger.info("total valid simple : " + str(len(data_list)))

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list[0:len(data_list) // 5 * 4], useHead=useHead,
                                                              modality_type=modality_type)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        valid_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list[len(data_list) // 5 * 4: -1], useHead=useHead,
                                                              modality_type=modality_type)
        valid_dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=True)

        for epoch in range(epoch_max_number):

            model_name = './models/audio24k_%d.model' % (epoch + 1)

            train(data_loader=train_dataloader, epoch=epoch)
            model.eval()
            valid(data_loader=valid_dataloader, epoch=epoch)
            model.train()

            if (epoch + 1) % 10 == 0:
                learning_rate = change_lr(optimizer,
                                          learning_rate)  # here the reference of learning_rate is also globally changed
            if (epoch + 1) % 5 == 0:
                torch.save(model, model_name)

    logger.info_loss()

def train_audio_OS_IS(data_loader, epoch):


    total_closs = 0.0
    total_rloss = 0.0
    total_running_loss = 0.0
    total_L1loss = 0.0
    total_L2loss = 0.0
    total_Beloss = 0.0
    start_time = time.time()

    for batch, (arr, reg_labels) in enumerate(data_loader):
        reg_labels = reg_labels.t()
        optimizer.zero_grad()
        model.train()
        el, nl, al, cl, ol, il = reg_labels
        cls_labels = []

        for item in reg_labels:
            cls_labels.append(torch.tensor(get_one_hot(item), dtype=torch.long))

        ecl, ncl, acl, ccl, ocl, icl = cls_labels
        x_cls, x_reg, _ = model(arr.cuda())
        # 预测的分类值
        ec = x_cls[0]
        nc = x_cls[1]
        ac = x_cls[2]
        cc = x_cls[3]
        oc = x_cls[4]
        ic = x_cls[5]

        # #logger.info(e.size(), n.size())
        # p_e = ec.max(1, keepdim=True)[1]
        # p_n = nc.max(1, keepdim=True)[1]
        # p_a = ac.max(1, keepdim=True)[1]
        # p_c = cc.max(1, keepdim=True)[1]
        # p_o = oc.max(1, keepdim=True)[1]
        # p_i = ic.max(1, keepdim=True)[1]

        # 计算分类损失
        lc_e = loss(ec, ecl.cuda())
        lc_n = loss(nc, ncl.cuda())
        lc_a = loss(ac, acl.cuda())
        lc_c = loss(cc, ccl.cuda())
        lc_o = loss(oc, ocl.cuda())
        lc_i = loss(ic, icl.cuda())
        # 真实值
        e = x_reg[0]
        n = x_reg[1]
        a = x_reg[2]
        c = x_reg[3]
        o = x_reg[4]
        i = x_reg[5]

        # 计算回归损失
        loss_e = L2loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                 L1loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                 bellloss(torch.mul(e, 100), el.float().view_as(e).cuda())

        loss_n = L2loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                 L1loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                 bellloss(torch.mul(n, 100), nl.float().view_as(n).cuda())

        loss_a = L2loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                 L1loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                 bellloss(torch.mul(a, 100), al.float().view_as(a).cuda())

        loss_c = L2loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                 L1loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                 bellloss(torch.mul(c, 100), cl.float().view_as(c).cuda())

        loss_o = L2loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                 L1loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                 bellloss(torch.mul(o, 100), ol.float().view_as(o).cuda())

        loss_i = L2loss(torch.mul(i, 100), il.float().view_as(i).cuda()) + \
                 L1loss(torch.mul(i, 100), il.float().view_as(i).cuda()) + \
                 bellloss(torch.mul(i, 100), il.float().view_as(i).cuda())

        # 分类损失
        total_L1loss = total_L1loss + \
                       (L1loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                        L1loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                        L1loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                        L1loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                        L1loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                        L1loss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size
        total_L2loss = total_L1loss + \
                       (L2loss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                        L2loss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                        L2loss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                        L2loss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                        L2loss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                        L2loss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size
        total_Beloss = total_L1loss + \
                       (bellloss(torch.mul(e, 100), el.float().view_as(e).cuda()) + \
                        bellloss(torch.mul(n, 100), nl.float().view_as(n).cuda()) + \
                        bellloss(torch.mul(a, 100), al.float().view_as(a).cuda()) + \
                        bellloss(torch.mul(c, 100), cl.float().view_as(c).cuda()) + \
                        bellloss(torch.mul(o, 100), ol.float().view_as(o).cuda()) + \
                        bellloss(torch.mul(i, 100), el.float().view_as(i).cuda())).item() / train_batch_size

        cls_loss = lc_e + lc_n + lc_a + lc_c + lc_o + lc_i

        # 回归损失
        reg_loss = loss_e + loss_n + loss_a + loss_c + loss_o + loss_i
        # 总损失

        if without_closs:
            running_loss = reg_loss
        else:
            # running_loss = cls_loss * 200 / (epoch+1) + reg_loss
            running_loss = cls_loss * 200 + reg_loss

        # running_loss = cls_loss

        running_loss.backward()
        optimizer.step()

        total_closs = total_closs + cls_loss.item() / train_batch_size
        total_rloss = total_rloss + reg_loss.item() / train_batch_size
        total_running_loss = total_running_loss + running_loss.item() / train_batch_size

        # logger.info('Training [current epoch: %2d total: %2d iter/all %4d/%4d]  Loss: %.4f \t time: %.3f @ %s' % (epoch, epoch_max_number, batch,
        #                                                                len(data_loader.dataset) / train_batch_size,
        #                                                                 running_loss.item(), time.time() - start_time,
        #                                                                     time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
        # logger.info('batch total loss:  %4f  %4f  %4f ' % (cls_loss.item() ,reg_loss.item(), running_loss.item()))
        # logger.info('%')

    train_plot_arr_allloss.append([total_rloss, total_closs, total_rloss, total_L1loss, total_L2loss, total_Beloss])

    if (epoch + 1) % 10 == 0:
        np.savetxt('train_plot_arr_allloss' + str(useHead) + '.csv', train_plot_arr_allloss,
                   delimiter=',', fmt='%.5f')

    logger.info("epoch: " + str(epoch + 1))
    logger.info("=============training=====================")
    logger.info('total allLoss:  %4f  total rloss:  %4f   total closs:  %4f ' % (
    total_running_loss, total_rloss, total_closs,))






def print_loss():

    train_plot_arr_allloss = np.loadtxt('train_plot_arr_allloss' + str(useHead) + '.csv', delimiter=',')
    valid_plot_arr_allloss = np.loadtxt('valid_plot_arr_allloss' + str(useHead) + '.csv', delimiter=',')

    plt.figure(figsize=(10,10))

    plt.subplot(231)
    plt.plot(train_plot_arr_allloss[:, 0], label='plot_arr_allloss')
    plt.plot(valid_plot_arr_allloss[:, 0], label='plot_arr_allloss_valid')

    plt.legend()


    plt.subplot(232)
    plt.plot(train_plot_arr_allloss[:, 1], label='plot_arr_closs')
    plt.plot(valid_plot_arr_allloss[:, 1], label='plot_arr_closs_valid')
    plt.legend()

    plt.subplot(233)
    plt.plot(train_plot_arr_allloss[:, 2], label='plot_arr_rloss')
    plt.plot(valid_plot_arr_allloss[:, 2], label='plot_arr_rloss_valid')
    plt.legend()

    plt.subplot(234)
    plt.plot(train_plot_arr_allloss[:, 3], label='total_L1loss')
    plt.plot(valid_plot_arr_allloss[:, 3], label='total_L1loss_valid')
    plt.legend()

    plt.subplot(235)
    plt.plot(train_plot_arr_allloss[:, 4], label='total_L2loss')
    plt.plot(valid_plot_arr_allloss[:, 4], label='total_L2loss_valid')
    plt.legend()

    plt.subplot(236)
    plt.plot(train_plot_arr_allloss[:, 5], label='total_Beloss')
    plt.plot(valid_plot_arr_allloss[:, 5], label='total_Beloss_valid')
    plt.legend()



    plt.xlabel('Epoch')
    plt.ylabel('Loss ')

    plt.savefig('Loss.png')
    plt.show()



if __name__ == '__main__':
    #


    train_image()


    # train_image_step1()
    #
    # model_name = "./models/step1_pre_useSence_50.model"
    #
    # model = torch.load(model_name).to("cuda:1")
    #
    # train_image()
    #
    # logger.info(train_stage, div_arr, modality_type, resnet_pretrained)
    #
    # if modality_type == 2:
    #     test_audio_resnet()
    # else:
    #     test_visual(epoch_num=30)
    #     extra_tree_regress_eval()
    #
    #     test_visual(epoch_num=40)
    #     extra_tree_regress_eval()
    #
    #     test_visual(epoch_num=5.0)
    #     extra_tree_regress_eval()


    #===============================EDA==========================================================

    # typ = "test"
    # with open('./dataset/%s/annotation_%s.pkl' % (typ, typ), 'rb') as fo:  # 读取pkl文件数据
    #     r = pickle.load(fo, encoding='bytes')

    # tmp1 = 0
    # tmp2 = 0
    # tmp3 = 0
    # tmp4 = 0
    # for item in r.values():
    #     for item2 in item.values():
    #         if item2 < 0.5:
    #             tmp1 = tmp1 + 1
    #         elif item2 < 0.6:
    #             tmp2 = tmp2 + 1
    #         elif item2 < 0.7:
    #             tmp3 = tmp3 + 1
    #         else:
    #             tmp4 = tmp4 + 1
    # logger.info(tmp1, tmp2, tmp3, tmp4)

    # dataType = "test"
    #
    # DATASET_PATH = "./dataset/" + dataType + "/videos/"
    # file_names = os.listdir(DATASET_PATH)
    #
    # file_list_test = [os.path.join(
    #     DATASET_PATH, file) for file in file_names]
    # #file_list.sort()
    #
    # exist_file = os.listdir("./dataset/" + dataType + "/audio/")
    # exist_file_name = []
    #
    # for item in exist_file:
    #     exist_file_name.append(DATASET_PATH + item[:-4] + ".mp4")
    #
    #
    # total_name =  file_list_test
    # file_list_test = []
    # for item in total_name:
    #     if item in exist_file_name:
    #         continue
    #     else:
    #         file_list_test.append(item)
    #
    # logger.info(len(file_list_test))
    #
    #
    #
    # for filename in file_list_test:
    #
    #
    #     #tmp = filename[0][:-4]
    #     #  index indicates the order in the list and [0] means its address, see the tuple example above.
    #     arr = []
    #
    #     y, sr = librosa.load(filename, sr=16000)
    #     if len(y) < 244832:
    #         tmp = np.zeros(244832 - len(y))
    #         tmp2 = y
    #
    #         y = np.append(tmp2, tmp)
    #     y = y.astype(np.float32)
    #
    #     DATASET_PATH = "./dataset/" + dataType + "/audio/"
    #
    #     tmp = DATASET_PATH + filename[len(DATASET_PATH)+1:-4] + ".npy"
    #
    #     np.save(tmp, y)
    #     logger.info(tmp)