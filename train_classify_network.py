import PersonalityDataset
from torch.utils.data import TensorDataset
from PersonalityDataset import get_data_list_and_big_five,get_one_hot
import new_resnet_6d
import torch
from new_resnet_6d import get_parameter_number
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from new_resnet_6d import ResNet_audio, ResNet_old_pretrain
import time
from BellLoss import BellingLoss
from config import useHead, without_closs,modality_type, resnet_pretrained,full_train,div_arr,device_num,random_seclect
from valid_resnet import test_audio_resnet ,test_visual,extra_tree_regress_eval
import pickle
import sys, os,random
import argparse
import  csv
import numpy as np


import logging

# 1.显示创建
logging.basicConfig(filename='step1_logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 2.定义logger,设定setLevel，FileHandler，setFormatter
logger = logging.getLogger(__name__)  # 定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
logger.setLevel(level=logging.INFO)  # 定义过滤级别
filehandler = logging.FileHandler("step1_log" + "_modality_type_" + str(modality_type) +  ".txt")  # Handler用于将日志记录发送至合适的目的地，如文件、终端等
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler()  # 日志信息显示在终端terminal
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)



BAST_RESULT = 1000000
# 训练参数

epoch_max_number = 50

train_batch_size = 30

learning_rate = 0.001

momentum = 0.9
num_classes = 4
weight_decay=0.005
torch.cuda.set_device(device_num)

# 模型建立

if modality_type != 2:
    model_classify = new_resnet_6d.resnet34_old_classify(pretrained=True, num_output=num_classes)
else:
    model_classify = new_resnet_6d.resnet34_audio_classify(pretrained=False, num_output=num_classes)

logger.info(get_parameter_number(model_classify))

model_classify  = model_classify.cuda()

logger.info(model_classify)

loss = torch.nn.CrossEntropyLoss()

train_plot_arr_allloss = []
valid_plot_arr_allloss = []
test_plot_arr_allloss = []



optimizer_class = torch.optim.SGD(model_classify.parameters(), lr=learning_rate , momentum=momentum, weight_decay=weight_decay)

data_list = get_data_list_and_big_five('./dataset/train', 'train')
random.shuffle (data_list )
data_list_train = data_list[0: 5400]
data_list_valid = data_list[5400: ]
test_data_list = get_data_list_and_big_five('./dataset/test', 'test')

# data_list = data_list[0:500]
# test_data_list = test_data_list[0:100]


logger.info({"BAST_RESULT":BAST_RESULT,
            "random_seclect":random_seclect,
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



def train_step1(data_loader, epoch):


    # model_classify  只有分类部分的  CR-net

    total_closs = 0.0



    for batch, (arr, choice_frame, name, cls_labels, reg_labels) in enumerate(data_loader):

        optimizer_class.zero_grad()
        model_classify.train()

        ecl, ncl, acl, ccl, ocl, icl = cls_labels

        x_cls = model_classify(arr.cuda())

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

        cls_loss = lc_e + lc_n + lc_a + lc_c + lc_o + lc_i

        cls_loss.backward()

        total_closs = total_closs + cls_loss.item()/train_batch_size

        optimizer_class.step()




    train_plot_arr_allloss.append([total_closs,])

    if (epoch + 1)%10  == 0:

        np.savetxt('class_train_plot_arr_allloss' + str(useHead) + '.csv',train_plot_arr_allloss,
                   delimiter=',', fmt='%.5f')


    logger.info("epoch: " + str(epoch + 1))
    logger.info ( "=============training=====================")
    logger.info('total closs:  %4f ' % ( total_closs,))

def valid_step1(data_loader, epoch):


    # model_classify  只有分类部分的  CR-net

    total_closs = 0.0

    with torch.no_grad():

        for batch, (arr, choice_frame, name, cls_labels, reg_labels) in enumerate(data_loader):
            model_classify.eval()

            ecl, ncl, acl, ccl, ocl, icl = cls_labels
            x_cls = model_classify(arr.cuda())

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


            cls_loss = lc_e + lc_n + lc_a + lc_c + lc_o + lc_i


            total_closs = total_closs + cls_loss.item()/train_batch_size


        valid_plot_arr_allloss.append([total_closs,])

        if (epoch + 1)%10  == 0:

            np.savetxt('class_valid_plot_arr_allloss' + str(useHead) + '.csv',valid_plot_arr_allloss,
                       delimiter=',', fmt='%.5f')



        logger.info ( "=============Validing=====================")
        logger.info('total closs:  %4f \n\n' % ( total_closs,))


def test_step1(data_loader, epoch):


    # model_classify  只有分类部分的  CR-net

    total_closs = 0.0




    with torch.no_grad():


        for batch, (arr, choice_frame, name, cls_labels, reg_labels) in enumerate(data_loader):
            model_classify.eval()

            ecl, ncl, acl, ccl, ocl, icl = cls_labels
            x_cls = model_classify(arr.cuda())

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

            cls_loss = lc_e + lc_n + lc_a + lc_c + lc_o + lc_i

            total_closs = total_closs + cls_loss.item()/train_batch_size




        global BAST_RESULT

        if BAST_RESULT > total_closs:
            BAST_RESULT = total_closs
            model_name = './models/class_best_shit_' + str(BAST_RESULT)[0:7] + "_modality_type_" + str(modality_type) +'.pt'

            logger.info("save the shit:  "+ str(total_closs))
            torch.save(model_classify.state_dict(), model_name)

        logger.info("epoch: " + str(epoch + 1))
        logger.info ( "=============Testing=====================")
        logger.info('total closs:  %4f ' % ( total_closs,))

        test_plot_arr_allloss.append([total_closs,])

    if (epoch + 1)%10  == 0:

        np.savetxt('class_test_plot_arr_allloss' + str(useHead) + '.csv',test_plot_arr_allloss,
                   delimiter=',', fmt='%.5f')

def change_lr(opt, learning_rate_tmp):
    learning_rate_tmp *= 0.1
    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate_tmp

    return learning_rate_tmp

def train_image_classify():

    if full_train:

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list, useHead=useHead,
                                                              modality_type=modality_type)
    else:

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list_train, useHead=useHead,
                                                              modality_type=modality_type)
        valid_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list_valid, useHead=useHead,
                                                              modality_type=modality_type)

        valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
        logger.info("valid simple:  " + str(len(valid_dataset.data_list)))



    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         test_data_list, useHead=useHead,
                                                          modality_type=modality_type)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)



    logger.info("train simple:  " + str(len(train_dataset.data_list)))

    logger.info("test simple:  " + str(len(test_dataset.data_list)))


    for epoch in range(epoch_max_number):

        global learning_rate

        if useHead:
            model_name = './models/step1_pre_useHead_%d.pt' % (epoch + 1)
        else:
            model_name = './models/step1_pre_useSence_%d.pt' % (epoch + 1)

        train_step1(data_loader=train_dataloader, epoch=epoch)

        if not full_train:
            valid_step1(data_loader=valid_dataloader, epoch=epoch)

        test_step1(data_loader=test_dataloader, epoch=epoch)

        if (epoch + 1) % 10 == 0:

            learning_rate = change_lr(optimizer_class,learning_rate)  # here the reference of learning_rate is also globally changed

            logger.info("======================" + str(learning_rate) + "========================")


    logger.info(model_name)


def print_loss():

    train_plot_arr_allloss = np.loadtxt('class_train_plot_arr_allloss' + str(useHead) + '.csv', delimiter=',')
    #valid_plot_arr_allloss = np.loadtxt('class_valid_plot_arr_allloss' + str(useHead) + '.csv', delimiter=',')
    test_plot_arr_allloss = np.loadtxt('class_test_plot_arr_allloss' + str(useHead) + '.csv', delimiter=',')

    plt.subplot(311)
    plt.plot(train_plot_arr_allloss, label='plot_arr_closs')
    #plt.plot(valid_plot_arr_allloss, label='plot_arr_closs_valid')

    plt.subplot(312)

    #plt.plot(valid_plot_arr_allloss, label='valid_plot_arr_allloss')

    plt.legend()


    plt.subplot(313)

    plt.plot(test_plot_arr_allloss, label='plot_arr_closs_test')



    plt.xlabel('Epoch')
    plt.ylabel('Loss ')

    plt.savefig('Loss_classify' + "_modality_type_" + str(modality_type) +  '.png')
    plt.show()


if __name__ == '__main__':
    train_image_classify()
    print_loss()


