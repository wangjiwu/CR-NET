import PersonalityDataset
from PersonalityDataset import get_data_list_and_big_five,get_one_hot
from new_resnet_6d import ResNet_single, BasicBlock_audio,model_urls, resnet34_pretrain
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import time
import math
import sys, os
import numpy as np
from config import useHead,modality_type
from torch.utils.data import TensorDataset

torch.cuda.set_device(0)

test_batch_size = 100

true_label = []
predict_label = []
L1loss = torch.nn.L1Loss()

def test(data_loader, model, epoch, flag):


    acc = [0., 0., 0., 0., 0., 0]

    feature_vector = []
    label_vector = []
    pre_label_vector = []

    with torch.no_grad():

        t = time.time()
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
                label_vector.append([el[i].item()/100, nl[i].item()/100, al[i].item()/100, cl[i].item()/100, ol[i].item()/100, il[i].item()/100])
                pre_label_vector.append([x_reg[0][i].cpu().numpy(), x_reg[1][i].cpu().numpy(),x_reg[2][i].cpu().numpy(),x_reg[3][i].cpu().numpy(),x_reg[4][i].cpu().numpy(),x_reg[5][i].cpu().numpy()])
            # print(e.size(), n.size())
            p_e = ec.max(1, keepdim=True)[1]
            p_n = nc.max(1, keepdim=True)[1]
            p_a = ac.max(1, keepdim=True)[1]
            p_c = cc.max(1, keepdim=True)[1]
            p_o = oc.max(1, keepdim=True)[1]
            p_i = ic.max(1, keepdim=True)[1]

            print('extr. cls. prediction data vs label ', p_e.cpu().data.view(1, -1), ecl.cpu().data)

            print('extr. prediction data vs label')
            np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
            print(torch.mul(x_reg[1], 100).squeeze(1).cpu().data.numpy())
            print(nl.float().data.numpy())
            print("\n\n")

            print('extr. reg. prediction data vs label ', x_reg[0].cpu().data.view(1, -1),
                  torch.mul(el, 0.01).cpu().data)

            for i in range(0, 6):
                acc[i] += torch.sum(torch.div(
                    torch.abs((x_reg[i].cpu().data - torch.mul(labels[i], 0.01).float().view_as(x_reg[i]))),
                    1
                )).item()

            print('%d batch e is' % (batch), 1 - acc[0] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch n is' % (batch), 1 - acc[1] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch a is' % (batch), 1 - acc[2] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch c is' % (batch), 1 - acc[3] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch o is' % (batch), 1 - acc[4] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch i is' % (batch), 1 - acc[5] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)

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

        #print(np.mean(error))
        result.append(np.mean(error))

    print(result)
    print(1 - np.mean(result[0:5]))
    print(result[5])

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

    # train_feature_vector = train_feature_vector_0 * 0.7  +  train_feature_vector_1 * 0.5
    # train_label_vector =  train_label_vector_0
    # test_feature_vector = test_feature_vector_0 * 0.7  + test_feature_vector_1 * 0.5
    # test_label_vector = test_label_vector_0

    if modality_type == 0:
        train_feature_vector = train_feature_vector_0 * 1
        train_label_vector =  train_label_vector_0 * 1
        test_feature_vector = test_feature_vector_0 * 1
        test_label_vector = test_label_vector_0 * 1

    elif modality_type == 1:
        train_feature_vector = train_feature_vector_1 * 1
        train_label_vector = train_label_vector_1 * 1
        test_feature_vector = test_feature_vector_1 * 1
        test_label_vector = test_label_vector_1 * 1
    elif modality_type == 2:
        train_feature_vector = train_feature_vector_2 * 1
        train_label_vector = train_label_vector_2 * 1
        test_feature_vector = test_feature_vector_2 * 1
        test_label_vector = test_label_vector_2 * 1

    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

    from sklearn.preprocessing import StandardScaler

    print("train_feature_arr:       " + str(len(train_feature_vector)))
    print("train_feature_score:     " + str(len(train_label_vector)))
    print("test_feature_arr:        " + str(len(test_feature_vector)))
    print("test_feature_score:      " + str(len(test_label_vector)))


    result_list = []

    for i in range(6):

        ss_x = StandardScaler()

        x_train = ss_x.fit_transform(train_feature_vector[:,:,i])
        x_test = ss_x.transform(test_feature_vector[:,:,i])

        y_train = train_label_vector[:,i]
        y_test =  test_label_vector[:,i]




        etr = ExtraTreesRegressor(n_estimators=200, max_features = 512)
        etr.fit(x_train, y_train)

        etr_y_predict = etr.predict(x_test)

        result = 0

        for i in range(len(etr_y_predict)):
            result = result + abs(etr_y_predict[i] - y_test[i])

        result =  result / len(etr_y_predict)

        print("extra tree eval: "  + str(1 - result))

        result_list.append(1 - result)
    print(result_list)
    print(np.mean(result_list[0:5]))
    print(result_list[5])

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

    # train_feature_vector = train_feature_vector_0 * 0.7 + train_feature_vector_2 * 0.3
    # train_label_vector =  train_label_vector_0 * 0.7 + train_label_vector_2 * 0.3
    # test_feature_vector = test_feature_vector_0 * 0.7 + test_feature_vector_2 * 0.3
    # test_label_vector = test_label_vector_0 * 0.7 + test_label_vector_2 * 0.3




    # train_feature_vector = np.concatenate([train_feature_vector_0, train_feature_vector_1], axis=1)
    # test_feature_vector = np.concatenate([test_feature_vector_0, test_feature_vector_1], axis=1)

    if modality_type == 0:
        train_feature_vector = train_feature_vector_0
        test_feature_vector = test_feature_vector_0
        train_label_vector =  train_label_vector_0
        test_label_vector = test_label_vector_0

    elif modality_type == 1:
        train_feature_vector = train_feature_vector_1
        test_feature_vector = test_feature_vector_1
        train_label_vector = train_label_vector_1
        test_label_vector = test_label_vector_1

    elif modality_type == 2:
        train_feature_vector = train_feature_vector_2
        test_feature_vector = test_feature_vector_2
        train_label_vector = train_label_vector_2
        test_label_vector = test_label_vector_2



    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

    from sklearn.preprocessing import StandardScaler

    print("train_feature_arr:       " + str(len(train_feature_vector)))
    print("train_feature_score:     " + str(len(train_label_vector)))
    print("test_feature_arr:        " + str(len(test_feature_vector)))
    print("test_feature_score:      " + str(len(test_label_vector)))


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




        etr = ExtraTreesRegressor(n_estimators=1000, max_features=6)
        etr.fit(x_train, y_train)

        etr_y_predict = etr.predict(x_test)

        result = 0

        for i in range(len(etr_y_predict)):
            result = result + abs(etr_y_predict[i] - y_test[i])

        result =  result / len(etr_y_predict)

        print("extra tree eval: "  + str(1 - result))

        result_list.append(1 - result)

    print(result_list)
    print(np.mean(result_list[0:5]))
    print(result_list[5])

def test_visual ():

    epoch_max_number = 1

    data_list = get_data_list_and_big_five('./dataset/train', 'train')


    train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                          data_list, useHead=useHead,
                                                          modality_type=modality_type)
    train_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)

    data_list = get_data_list_and_big_five('./dataset/test', 'test')
    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         data_list, useHead=useHead,
                                                         modality_type=modality_type)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    epoch_num = 30

    for epoch in range(epoch_max_number):

        if useHead:

            model_name = './models/6000_pre_useHead_' + str(epoch_num) + '.model'
        else:
           #model_name = './models/pre_useSence_' + str(epoch_num) + '-0.905.model'
            model_name = './models/pre_useSence_' + str(epoch_num) + '.model'

        model = torch.load(model_name).to("cuda:0")

        model.eval()
        test(test_dataloader,model , epoch, "test")

        test(train_dataloader, model , epoch, "train")

    #extra_tree_regress_eval()

def test_audio_eval(data_loader, model, epoch, flag):
    print('test')

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
            # print(e.size(), n.size())
            p_e = ec.max(1, keepdim=True)[1]
            p_n = nc.max(1, keepdim=True)[1]
            p_a = ac.max(1, keepdim=True)[1]
            p_c = cc.max(1, keepdim=True)[1]
            p_o = oc.max(1, keepdim=True)[1]
            p_i = ic.max(1, keepdim=True)[1]

            print('extr. cls. prediction data vs label ', p_e.cpu().data.view(1, -1), ecl.cpu().data)

            print('extr. prediction data vs label')
            np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
            print(torch.mul(x_reg[1], 100).squeeze(1).cpu().data.numpy())
            print(nl.float().data.numpy())
            print("\n\n")

            print('extr. reg. prediction data vs label ', x_reg[0].cpu().data.view(1, -1),
                  torch.mul(el, 0.01).cpu().data)

            for i in range(0, 6):
                acc[i] += torch.sum(torch.div(
                    torch.abs((x_reg[i].cpu().data - torch.mul(reg_labels[i], 0.01).float().view_as(x_reg[i]))),
                    1
                )).item()

            print('%d batch e is' % (batch), 1 - acc[0] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch n is' % (batch), 1 - acc[1] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch a is' % (batch), 1 - acc[2] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch c is' % (batch), 1 - acc[3] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch o is' % (batch), 1 - acc[4] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)
            print('%d batch i is' % (batch), 1 - acc[5] / ((batch + 1) * test_batch_size), 'time ', time.time() - t)

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

        print(np.mean(error))

def test_audio():
    import scipy.io as scio
    train_batch_size = 128
    epoch_max_number = 1
    path = './feature_extraction_result/OS_IS13.mat'

    data = scio.loadmat(path)

    # 读取数据 和训练集视频对应 6000个
    #print (data["OS_IS13"][0][0][0])

    dict_audio_data = {}

    for item in  range(8000):
        file_name = data["OS_IS13"][0][0][0][item][0][0]
        # print(file_name)
        # print(data["OS_IS13"][0][0][1][item])
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

    epoch_num = 30

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

def test_audio_resnet ():
    epoch_max_number = 1

    data_list = get_data_list_and_big_five('./dataset/train', 'train')
    train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                          data_list, useHead=useHead,
                                                          modality_type=modality_type)
    train_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)

    data_list = get_data_list_and_big_five('./dataset/test', 'test')
    test_dataset = PersonalityDataset.PersonalityDataset('./dataset/test', 'test',
                                                         data_list, useHead=useHead,
                                                         modality_type=modality_type)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    epoch_num = 50

    for epoch in range(epoch_max_number):


        model_name = './models/audio24k_' + str(epoch_num) + '.model'

        model = torch.load(model_name).to("cuda:0")

        model.eval()
        test(test_dataloader,model , epoch, "test")

        test(train_dataloader, model , epoch, "train")

    #extra_tree_regress_eval()

def extre_resnet_image ():
    test_batch_size = 1  # 20
    epoch_max_number = 1

    # data_list = get_data_list_and_big_five('./dataset/train', 'train')
    # train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
    #                                                       data_list, useHead=False,
    #                                                       modality_type=0)
    # train_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)



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
    #         print(count)
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
            print(count)
            count  = count + 1












if __name__ == '__main__':
    # #test_audio()
    print('test type:' + str(modality_type))

    if modality_type == 2 :
        test_audio_resnet()
    else:
        test_visual()

    extra_tree_regress_eval()
    # rf_score_regress_eval()

    #extre_resnet_image()

