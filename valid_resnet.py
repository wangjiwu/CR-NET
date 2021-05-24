import PersonalityDataset
from PersonalityDataset import get_data_list_and_big_five,get_one_hot
from new_resnet_6d import ResNet_single, BasicBlock_audio,model_urls, resnet34_pretrain
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import time
import math
import sys, os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

import numpy as np
from config import useHead,modality_type, resnet_pretrained, full_train, div_arr, device_num
from torch.utils.data import TensorDataset

torch.cuda.set_device(device_num)

test_batch_size = 100

true_label = []
predict_label = []
L1loss = torch.nn.L1Loss()

import logging

# 1.显示创建
logging.basicConfig(filename='valid_logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 2.定义logger,设定setLevel，FileHandler，setFormatter
logger = logging.getLogger(__name__)  # 定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
logger.setLevel(level=logging.INFO)  # 定义过滤级别
filehandler = logging.FileHandler("valid_log.txt")  # Handler用于将日志记录发送至合适的目的地，如文件、终端等
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler()  # 日志信息显示在终端terminal
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)



def test(data_loader, model, epoch, flag):


    acc = [0., 0., 0., 0., 0., 0]

    feature_vector = []
    label_vector = []
    pre_label_vector = []
    model.eval()

    with torch.no_grad():
        for batch, (arr, choice_frame, name, cls_labels, labels) in enumerate(data_loader):
            model.eval()

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
                label_vector.append([el[i].item()/100, nl[i].item()/100, al[i].item()/100, cl[i].item()/100, ol[i].item()/100, il[i].item()/100])
                pre_label_vector.append([x_reg[0][i].cpu().numpy(), x_reg[1][i].cpu().numpy(),x_reg[2][i].cpu().numpy(),x_reg[3][i].cpu().numpy(),x_reg[4][i].cpu().numpy(),x_reg[5][i].cpu().numpy()])



            # logger.info('extr. prediction data vs label')
            # np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
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


    feature_vector = np.array(feature_vector)
    np.save("./fusion_feature/" + flag +'_feature_vector' + str(modality_type)+ '.npy', feature_vector)

    label_vector = np.array(label_vector)
    np.save("./fusion_feature/" + flag + '_label_vector' + str(modality_type)+ '.npy', label_vector)
    pre_label_vector = np.array(pre_label_vector)
    np.save("./fusion_feature/" + flag + '_pre_label_vector' + str(modality_type) + '.npy', pre_label_vector)


    result  = []

    for i in range (6):
        x = label_vector[:, i]
        y = pre_label_vector[:, i]
        error = []
        for j in range(len(x)):
            error.append(abs(x[j] - y[j]))

        #logger.info(np.mean(error))
        result.append(np.mean(error))




    logger.info(result)
    logger.info(1 - np.mean(result[0:5]))
    logger.info(result[5])

def extra_tree_regress_eval():

    train_feature_vector_0 = np.load("./fusion_feature/" + 'train_feature_vector' + str(0)+ '.npy',allow_pickle=True)
    train_label_vector_0 = np.load("./fusion_feature/" + 'train_label_vector' + str(0)+ '.npy')
    test_feature_vector_0 = np.load("./fusion_feature/" +'test_feature_vector' + str(0)+ '.npy',allow_pickle=True)
    test_label_vector_0 = np.load("./fusion_feature/" +'test_label_vector' + str(0)+ '.npy')

    train_feature_vector_1 = np.load("./fusion_feature/" +'train_feature_vector' + str(1)+ '.npy',allow_pickle=True)
    train_label_vector_1 = np.load("./fusion_feature/" +'train_label_vector' + str(1)+ '.npy')
    test_feature_vector_1 = np.load("./fusion_feature/" +'test_feature_vector' + str(1)+ '.npy',allow_pickle=True)
    test_label_vector_1 = np.load("./fusion_feature/" +'test_label_vector' + str(1)+ '.npy')

    train_feature_vector_2 = np.load("./fusion_feature/" +'train_feature_vector' + str(2)+ '.npy',allow_pickle=True)
    train_label_vector_2 = np.load("./fusion_feature/" +'train_label_vector' + str(2)+ '.npy')
    test_feature_vector_2 = np.load("./fusion_feature/" +'test_feature_vector' + str(2)+ '.npy',allow_pickle=True)
    test_label_vector_2 = np.load("./fusion_feature/" +'test_label_vector' + str(2)+ '.npy')



    # 简单融合  sence ： face ： audio = 7 ：5 ： 3

    # if modality_type == 0:
    #     train_feature_vector = train_feature_vector_0 * 1
    #     train_label_vector =  train_label_vector_0 * 1
    #     test_feature_vector = test_feature_vector_0 * 1
    #     test_label_vector = test_label_vector_0 * 1
    #
    # elif modality_type == 1:
    #     train_feature_vector = train_feature_vector_1 * 1
    #     train_label_vector = train_label_vector_1 * 1
    #     test_feature_vector = test_feature_vector_1 * 1
    #     test_label_vector = test_label_vector_1 * 1
    # elif modality_type == 2:
    #     train_feature_vector = train_feature_vector_2 * 1
    #     train_label_vector = train_label_vector_2 * 1
    #     test_feature_vector = test_feature_vector_2 * 1
    #     test_label_vector = test_label_vector_2 * 1

    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

    from sklearn.preprocessing import StandardScaler

    etr_y_predict_0 = []
    etr_y_predict_1 = []
    etr_y_predict_2 = []

    #================================================================



    # test_modality = "0"
    # logger.info("test_modality: " + test_modality)
    #
    # logger.info("train_feature_arr:       " + str(len(train_feature_vector)))
    # logger.info("train_feature_score:     " + str(len(train_label_vector)))
    # logger.info("test_feature_arr:        " + str(len(test_feature_vector)))
    # logger.info("test_feature_score:      " + str(len(test_label_vector)))

    result_list = []


    import  random

    for i in range(1):

        weight = [5, 7, 3]
        logger.info(weight)
        for i in range(6):

            ss_x = StandardScaler()

            x_train = ss_x.fit_transform(train_feature_vector_0[:,:,i]) * weight[0] \
                      + ss_x.fit_transform(train_feature_vector_1[:,:,i]) * weight[1]\
                      + ss_x.fit_transform(train_feature_vector_2[:,:,i])  * weight[2]

            x_train = ss_x.fit_transform(x_train)

            x_test = ss_x.fit_transform(test_feature_vector_0[:,:,i]) * weight[0] \
                      + ss_x.fit_transform(test_feature_vector_1[:,:,i]) * weight[1]\
                      + ss_x.fit_transform(test_feature_vector_2[:,:,i])  * weight[2]

            x_test = ss_x.fit_transform(x_test)

            y_train = train_label_vector_0[:,i]
            y_test =  test_label_vector_0[:,i]


            etr = ExtraTreesRegressor(n_estimators=200, max_features = 512)
            etr.fit(x_train, y_train)

            etr_y_predict = etr.predict(x_test)
            etr_y_predict_0.append(etr_y_predict)

            result = 0

            for i in range(len(etr_y_predict)):
                result = result + abs(etr_y_predict[i] - y_test[i])

            result =  result / len(etr_y_predict)

            logger.info("extra tree eval: "  + str(1 - result))

            result_list.append(1 - result)

        # etr_y_predict_0 = np.array(etr_y_predict_0)
        # etr_y_predict_0 = np.swapaxes(etr_y_predict_0, 1, 0)
        logger.info(result_list)
        logger.info(np.mean(result_list[0:5]))
        logger.info(result_list[5])

    # pre_label_vector = np.array(etr_y_predict_0)
    # np.save("./fusion_feature/" + 'etr_pre_score_' + str(0) + '.npy', pre_label_vector)
    #
    # #================================================================
    #
    #
    # train_label_vector = train_label_vector_1
    # train_feature_vector = train_feature_vector_1
    #
    # test_modality = "1"
    # logger.info("test_modality: " + test_modality)
    #
    # logger.info("train_feature_arr:       " + str(len(train_feature_vector)))
    # logger.info("train_feature_score:     " + str(len(train_label_vector)))
    # logger.info("test_feature_arr:        " + str(len(test_feature_vector)))
    # logger.info("test_feature_score:      " + str(len(test_label_vector)))
    #
    # result_list = []
    #
    # for i in range(6):
    #
    #     ss_x = StandardScaler()
    #
    #     x_train = ss_x.fit_transform(train_feature_vector[:, :, i])
    #     x_test = ss_x.transform(test_feature_vector[:, :, i])
    #
    #     y_train = train_label_vector[:, i]
    #     y_test = test_label_vector[:, i]
    #
    #     etr = ExtraTreesRegressor(n_estimators=200, max_features=512)
    #     etr.fit(x_train, y_train)
    #
    #     etr_y_predict = etr.predict(x_test)
    #     etr_y_predict_1.append(etr_y_predict)
    #     result = 0
    #
    #     for i in range(len(etr_y_predict)):
    #         result = result + abs(etr_y_predict[i] - y_test[i])
    #
    #     result = result / len(etr_y_predict)
    #
    #     logger.info("extra tree eval: " + str(1 - result))
    #
    #     result_list.append(1 - result)
    #
    # etr_y_predict_1 = np.array(etr_y_predict_1)
    # etr_y_predict_1 = np.swapaxes(etr_y_predict_1, 1, 0)
    # pre_label_vector = np.array(etr_y_predict_1)
    # np.save("./fusion_feature/" + 'etr_pre_score_' + str(1) + '.npy', pre_label_vector)
    #
    # logger.info(result_list)
    # logger.info(np.mean(result_list[0:5]))
    # logger.info(result_list[5])
    #
    #
    # #==============================================================
    #
    #
    # train_label_vector = train_label_vector_2
    # train_feature_vector = train_feature_vector_2
    #
    # test_modality = "2"
    # logger.info("test_modality: " + test_modality)
    #
    # logger.info("train_feature_arr:       " + str(len(train_feature_vector)))
    # logger.info("train_feature_score:     " + str(len(train_label_vector)))
    # logger.info("test_feature_arr:        " + str(len(test_feature_vector)))
    # logger.info("test_feature_score:      " + str(len(test_label_vector)))
    #
    # result_list = []
    #
    # for i in range(6):
    #
    #     ss_x = StandardScaler()
    #
    #     x_train = ss_x.fit_transform(train_feature_vector[:, :, i])
    #     x_test = ss_x.transform(test_feature_vector[:, :, i])
    #
    #     y_train = train_label_vector[:, i]
    #     y_test = test_label_vector[:, i]
    #
    #     etr = ExtraTreesRegressor(n_estimators=200, max_features=512)
    #     etr.fit(x_train, y_train)
    #
    #     etr_y_predict = etr.predict(x_test)
    #     etr_y_predict_2.append(etr_y_predict)
    #
    #     result = 0
    #
    #     for i in range(len(etr_y_predict)):
    #         result = result + abs(etr_y_predict[i] - y_test[i])
    #
    #     result = result / len(etr_y_predict)
    #
    #     logger.info("extra tree eval: " + str(1 - result))
    #
    #     result_list.append(1 - result)
    #
    # etr_y_predict_2 = np.array(etr_y_predict_2)
    # etr_y_predict_2 = np.swapaxes(etr_y_predict_2, 1, 0)
    # pre_label_vector = np.array(etr_y_predict_2)
    # np.save("./fusion_feature/" + 'etr_pre_score_' + str(2) + '.npy', pre_label_vector)
    #
    # logger.info(result_list)
    # logger.info(np.mean(result_list[0:5]))
    # logger.info(result_list[5])


def rf_score_regress_eval():

    train_feature_vector_0 = np.load("./fusion_feature/" + 'train_pre_label_vector' + str(0)+ '.npy',allow_pickle=True)
    train_label_vector_0 = np.load("./fusion_feature/" + 'train_label_vector' + str(0)+ '.npy')
    test_feature_vector_0 = np.load("./fusion_feature/" +'test_pre_label_vector' + str(0)+ '.npy',allow_pickle=True)
    test_label_vector_0 = np.load("./fusion_feature/" +'test_label_vector' + str(0)+ '.npy')

    train_feature_vector_1 = np.load("./fusion_feature/" + 'train_pre_label_vector' + str(1) + '.npy',
                                     allow_pickle=True)
    train_label_vector_1 = np.load("./fusion_feature/" + 'train_label_vector' + str(1) + '.npy')
    test_feature_vector_1 = np.load("./fusion_feature/" + 'test_pre_label_vector' + str(1) + '.npy',
                                    allow_pickle=True)
    test_label_vector_1 = np.load("./fusion_feature/" + 'test_label_vector' + str(1) + '.npy')


    train_feature_vector_2 = np.load("./fusion_feature/" +'train_pre_label_vector' + str(2)+ '.npy',allow_pickle=True)
    train_label_vector_2 = np.load("./fusion_feature/" +'train_label_vector' + str(2)+ '.npy')
    test_feature_vector_2 = np.load("./fusion_feature/" +'test_pre_label_vector' + str(2)+ '.npy',allow_pickle=True)
    test_label_vector_2 = np.load("./fusion_feature/" +'test_label_vector' + str(2)+ '.npy')

    # 简单融合  sence ： face ： audio = 7 ：5 ： 3

    # ss_x.fit_transform()

    train_feature_vector= train_feature_vector_0 * 0.7 + train_feature_vector_1 * 0.5 + train_feature_vector_2 * 0.3
    train_label_vector  =  train_label_vector_0 * 0.7 + train_label_vector_1 * 0.5 + train_label_vector_2 * 0.3
    test_feature_vector = test_feature_vector_0 * 0.7 + test_feature_vector_1 * 0.5 + test_feature_vector_2 * 0.3
    test_label_vector   = test_label_vector_0 * 0.7 + test_label_vector_1 * 0.5 + test_label_vector_2 * 0.3





    # if modality_type == 0:
    #     train_feature_vector = train_feature_vector_0
    #     test_feature_vector = test_feature_vector_0
    #     train_label_vector =  train_label_vector_0
    #     test_label_vector = test_label_vector_0
    #
    # elif modality_type == 1:
    #     train_feature_vector = train_feature_vector_1
    #     test_feature_vector = test_feature_vector_1
    #     train_label_vector = train_label_vector_1
    #     test_label_vector = test_label_vector_1
    #
    # elif modality_type == 2:
    #     train_feature_vector = train_feature_vector_2
    #     test_feature_vector = test_feature_vector_2
    #     train_label_vector = train_label_vector_2
    #     test_label_vector = test_label_vector_2



    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

    from sklearn.preprocessing import StandardScaler

    logger.info("train_feature_arr:       " + str(len(train_feature_vector)))
    logger.info("train_feature_score:     " + str(len(train_label_vector)))
    logger.info("test_feature_arr:        " + str(len(test_feature_vector)))
    logger.info("test_feature_score:      " + str(len(test_label_vector)))


    result_list = []

    train_feature_vector = np.array(train_feature_vector[:,:,0])
    test_feature_vector = np.array(test_feature_vector[:,:,0])

    ss_x = StandardScaler()

    x_train = ss_x.fit_transform(train_feature_vector)
    x_test = ss_x.transform(test_feature_vector)

    result_list = []

    for i in range(6):

        y_train = train_label_vector[:,i]
        y_test =  test_label_vector[:,i]




        etr = ExtraTreesRegressor(n_estimators=200, max_features=6)
        etr.fit(x_train, y_train)

        etr_y_predict = etr.predict(x_test)

        result = 0

        for i in range(len(etr_y_predict)):
            result = result + abs(etr_y_predict[i] - y_test[i])

        result =  result / len(etr_y_predict)

        logger.info("extra tree eval: "  + str(1 - result))

        result_list.append(1 - result)

    logger.info(result_list)
    logger.info(np.mean(result_list[0:5]))
    logger.info(result_list[5])

def test_visual (model_name, modality_type_test):

    epoch_max_number = 1
    if modality_type_test == 0:
        useHead = False
    elif modality_type_test == 1:
        useHead = True

    data_list = get_data_list_and_big_five('./dataset/train', 'train')


    train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                          data_list, useHead=useHead,
                                                          modality_type=modality_type_test)
    train_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)

    data_list = get_data_list_and_big_five('./dataset/test', 'test')
    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         data_list, useHead=useHead,
                                                         modality_type=modality_type_test)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)



    model = torch.load(model_name).to("cuda:2")



    model.eval()
    test(test_dataloader,model , 0, "test")

    test(train_dataloader, model , 0, "train")

    logger.info(model_name)

    #extra_tree_regress_eval()

def test_audio_eval(data_loader, model, epoch, flag):
    logger.info('test')

    acc = [0., 0., 0., 0., 0., 0]

    feature_vector = []
    label_vector = []
    pre_label_vector = []

    with torch.no_grad():

        t = time.time()
        for batch, (arr, reg_labels) in enumerate(data_loader):
            reg_labels = reg_labels.t()
            el, nl, al, cl, ol, il = reg_labels * 100
            cls_labels = []

            for item in reg_labels:
                cls_labels.append(torch.tensor(get_one_hot(item), dtype=torch.long))

            ecl, ncl, acl, ccl, ocl, icl = cls_labels
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
                label_vector.append([el[i].item()/100, nl[i].item()/100, al[i].item()/100, cl[i].item()/100, ol[i].item()/100, il[i].item()/100])
                pre_label_vector.append([x_reg[0][i].cpu().numpy(), x_reg[1][i].cpu().numpy(),x_reg[2][i].cpu().numpy(),x_reg[3][i].cpu().numpy(),x_reg[4][i].cpu().numpy(),x_reg[5][i].cpu().numpy()])
            # logger.info(e.size(), n.size())
            p_e = ec.max(1, keepdim=True)[1]
            p_n = nc.max(1, keepdim=True)[1]
            p_a = ac.max(1, keepdim=True)[1]
            p_c = cc.max(1, keepdim=True)[1]
            p_o = oc.max(1, keepdim=True)[1]
            p_i = ic.max(1, keepdim=True)[1]

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
                    torch.abs((x_reg[i].cpu().data - torch.mul(reg_labels[i], 0.01).float().view_as(x_reg[i]))),
                    1
                )).item()

            logger.info('%d batch e is' % (batch), 1 - acc[0] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            logger.info('%d batch n is' % (batch), 1 - acc[1] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            logger.info('%d batch a is' % (batch), 1 - acc[2] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            logger.info('%d batch c is' % (batch), 1 - acc[3] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            logger.info('%d batch o is' % (batch), 1 - acc[4] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            logger.info('%d batch i is' % (batch), 1 - acc[5] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)

    feature_vector = np.array(feature_vector)
    np.save(flag +'_feature_vector' + str(modality_type)+ '.npy', feature_vector)

    label_vector = np.array(label_vector)
    np.save( flag + '_label_vector' + str(modality_type)+ '.npy', label_vector)

    pre_label_vector = np.array(pre_label_vector)



    for i in range (6):
        x = label_vector[:, i]
        y = pre_label_vector[:, i]
        error = []
        for j in range(len(x)):
            error.append(abs(x[j] - y[j]))

        logger.info(np.mean(error))

def test_audio():
    import scipy.io as scio
    train_batch_size = 128
    epoch_max_number = 1
    path = './feature_extraction_result/OS_IS13.mat'

    data = scio.loadmat(path)

    # 读取数据 和训练集视频对应 6000个
    #logger.info (data["OS_IS13"][0][0][0])

    dict_audio_data = {}

    for item in  range(8000):
        file_name = data["OS_IS13"][0][0][0][item][0][0]
        # logger.info(file_name)
        # logger.info(data["OS_IS13"][0][0][1][item])
        dict_audio_data[file_name] = data["OS_IS13"][0][0][1][item]

    data_list = get_data_list_and_big_five('./dataset/train', 'train')
    train_feature = []
    train_label = []
    for item in data_list:
        train_feature.append(dict_audio_data[item[0]])
        train_label.append(item[1:])
    train_feature = torch.Tensor(train_feature)
    train_label = torch.Tensor(train_label)
    train_ids = TensorDataset(train_feature, train_label)
    train_dataloader = DataLoader(train_ids, batch_size=train_batch_size, shuffle=True)

    data_list = get_data_list_and_big_five('./dataset/test', 'test')
    test_feature = []
    test_label = []
    for item in data_list:
        test_feature.append(dict_audio_data[item[0]])
        test_label.append(item[1:])
    test_feature = torch.Tensor(test_feature)
    test_label = torch.Tensor(test_label)
    test_ids = TensorDataset(test_feature, test_label)
    test_dataloader = DataLoader(test_ids, batch_size=train_batch_size, shuffle=True)

    epoch_num = 40

    for epoch in range(epoch_max_number):

        if useHead:

            model_name = './models/audio_' + str(epoch_num) + '.model'
        else:
            model_name = './models/audio_' + str(epoch_num) + '.model'

        model = torch.load(model_name).to("cuda:0")

        model.eval()
        test_audio_eval(test_dataloader, model, epoch, "test")

        test_audio_eval(train_dataloader, model, epoch, "train")

    #extra_tree_regress_eval()

def test_audio_resnet (model_name):
    epoch_max_number = 1

    data_list = get_data_list_and_big_five('./dataset/train', 'train')
    train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                          data_list, useHead=useHead,
                                                          modality_type=2)
    train_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)

    data_list = get_data_list_and_big_five('./dataset/test', 'test')
    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         data_list, useHead=useHead,
                                                         modality_type=2)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


    for epoch in range(epoch_max_number):



        model = torch.load(model_name).to("cuda:2")

        model.eval()
        test(test_dataloader,model , 0, "test")

        test(train_dataloader, model , 0, "train")

    #extra_tree_regress_eval()

def extre_resnet_image ():
    test_batch_size = 1  # 20


    data_list = get_data_list_and_big_five('./dataset/test', 'test')
    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         data_list, useHead=useHead,
                                                         modality_type=modality_type)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


    model = resnet34_pretrain()
    model = model.cuda()
    model.eval()

    count = 0


    # with torch.no_grad():
    #
    #     t = time.time()
    #     for batch, (arr1,arr2, choice_frame, name, cls_labels, labels) in enumerate(train_dataloader):
    #         filename = choice_frame[0]
    #         x_feature_sence = model(arr1.cuda())
    #         x_feature_sence = x_feature_sence.cpu().numpy()
    #
    #         x_feature_face = model(arr2.cuda())
    #         x_feature_face = x_feature_face.cpu().numpy()
    #
    #         output = open("./dataset/train/preTrain_feature/" + filename[0][:-4] + ".pkl", "wb")
    #
    #         data = {}
    #         data["sence_series"] = x_feature_sence
    #         data["head_series"] = x_feature_face
    #
    #         pickle.dump(data, output)
    #         logger.info(count)
    #         count  = count + 1
    #


    with torch.no_grad():

        t = time.time()
        for batch, (arr1,arr2, choice_frame, name, cls_labels, labels) in enumerate(test_dataloader):
            filename = choice_frame[0]
            x_feature_sence = model(arr1.cuda())
            x_feature_sence = x_feature_sence.cpu().numpy()

            x_feature_face = model(arr2.cuda())
            x_feature_face = x_feature_face.cpu().numpy()

            output = open("./dataset/test/preTrain_feature/" + filename[0][:-4] + ".pkl", "wb")

            data = {}
            data["sence_series"] = x_feature_sence
            data["head_series"] = x_feature_face

            pickle.dump(data, output)
            logger.info(count)
            count  = count + 1












if __name__ == '__main__':
    # # #test_audio()

    logger.info({
                 "useHead": useHead,
                 "modality_type": modality_type,
                 "device_num": device_num,
                 "resnet_pretrained": resnet_pretrained,
                 "full_train": full_train,
                 "div_arr": div_arr,
                 })

    # test_audio_resnet("./models/best_shit_0.89639_modality_type_2.model")
    # test_visual("./models/best_shit_0.91438.model", 1)
    test_visual("./models/best_shit_0.91160_modality_type_0.model", 0)

    #rf_score_regress_eval()
    extra_tree_regress_eval()

    #
    # test_label_vector_0 = np.load("./fusion_feature/" +'etr_pre_score_' + str(0)+ '.npy')
    # test_label_vector_1 = np.load("./fusion_feature/" +'etr_pre_score_' + str(1)+ '.npy')
    # test_label_vector_2 = np.load("./fusion_feature/" +'etr_pre_score_' + str(2)+ '.npy')
    #
    # result_list = []
    #
    # weight = [0.7,0.5,0.3]
    #
    # etr_y_predict = test_label_vector_0 * weight[0] + test_label_vector_1 *  weight[1]  + test_label_vector_2 * weight[2]
    #
    # etr_y_predict = etr_y_predict / sum(weight)
    #
    # # etr_y_predict = test_label_vector_2
    #
    # etr_y_predict = np.squeeze(etr_y_predict)
    #
    # y_test = np.load("./fusion_feature/" +'test_label_vector' + str(0)+ '.npy')
    #
    # result = 0
    #
    # for i in range(len(etr_y_predict)):
    #     result = result + abs(etr_y_predict[i] - y_test[i])
    #
    # result = result / len(etr_y_predict)
    #
    # logger.info("extra tree eval: " + str(1 - result))
    #
    # result_list = 1 - result
    # logger.info(result_list)
    # logger.info(np.mean(result_list[0:5]))
    # logger.info(result_list[5])


