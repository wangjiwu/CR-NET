import PersonalityDataset
from torch.utils.data import TensorDataset
import librosa
from PersonalityDataset import get_data_list_and_big_five,get_one_hot
import new_resnet_6d
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from new_resnet_6d import ResNet_audio, ResNet_old_pretrain
import time
from BellLoss import BellingLoss
from config import useHead, without_closs,modality_type, resnet_pretrained,full_train,div_arr
import pickle
import sys, os,random
import argparse
import  csv
import numpy as np




epoch_max_number =30

train_batch_size = 32

#learning_rate = 0.00002

learning_rate = 0.00002


momentum = 0.9
num_classes = 4
weight_decay = 0.005
torch.cuda.set_device(0)





train_plot_arr_allloss = []
valid_plot_arr_allloss = []


if modality_type != 2:

    model = new_resnet_6d.resnet34_old(pretrained=resnet_pretrained,num_output=4)

    #model = ResNet_old_pretrain()

else :

    model = new_resnet_6d.resnet34_audio(pretrained=resnet_pretrained,num_output=4)


model = model.cuda()
print("modality_type: " + str(modality_type))
print("useHead? :" +  str(useHead))


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

# optimizer = torch.optim.Adam([{'params': base_params},
#                               {'params': model.featconv_e.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.featconv_n.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.featconv_a.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.featconv_c.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.featconv_o.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.featconv_i.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.rfc_e.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.rfc_n.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.rfc_a.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.rfc_c.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.rfc_o.parameters(), 'lr': learning_rate * 100},
#                               {'params': model.rfc_i.parameters(), 'lr': learning_rate * 100}],
#                              lr=learning_rate,
#                              betas=[0.5, 0.999], weight_decay=weight_decay)



# optimizer_classify = torch.optim.SGD([
#                              {'params': model.fc_e.parameters(), 'lr': learning_rate * 100},
#                              {'params': model.fc_n.parameters(), 'lr': learning_rate * 100},
#                              {'params': model.fc_a.parameters(), 'lr': learning_rate * 100},
#                              {'params': model.fc_c.parameters(), 'lr': learning_rate * 100},
#                              {'params': model.fc_o.parameters(), 'lr': learning_rate * 100},
#                              {'params': model.fc_i.parameters(), 'lr': learning_rate * 100}],
#                              lr=learning_rate, momentum=momentum,
#                              weight_decay=weight_decay)


#  typ = args.type

#  data preparation


data_list = get_data_list_and_big_five('./dataset/train', 'train')
#data_list = data_list[0:200]
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

        optimizer.zero_grad()


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

        #cls_loss.backward(retain_graph=True)
        #optimizer_classify.step()

        # 回归损失
        reg_loss = loss_e + loss_n + loss_a + loss_c + loss_o + loss_i
        # 总损失

        if without_closs:
            running_loss =  reg_loss
        else:
            running_loss = cls_loss * epoch_max_number * 4 / (epoch+1) + reg_loss
            #running_loss = cls_loss * 200  + reg_loss

        # running_loss = cls_loss

        running_loss.backward()
        optimizer.step()

        total_closs = total_closs + cls_loss.item()/train_batch_size
        total_rloss = total_rloss + reg_loss.item() / train_batch_size
        total_running_loss = total_running_loss + running_loss.item() / train_batch_size




    train_plot_arr_allloss.append([total_rloss, total_closs, total_rloss, total_L1loss, total_L2loss, total_Beloss])

    if (epoch + 1)%10  == 0:

        np.savetxt('train_plot_arr_allloss' + str(useHead) + '.csv',train_plot_arr_allloss,
                   delimiter=',', fmt='%.5f')

    print("epoch: " + str(epoch + 1))
    print ( "=============training=====================")
    print('total allLoss:  %4f  total rloss:  %4f   total closs:  %4f ' % (total_running_loss, total_rloss, total_closs,))

def valid(data_loader, epoch):


    total_closs = 0.0
    total_rloss = 0.0
    total_running_loss = 0.0
    total_L1loss = 0.0
    total_L2loss = 0.0
    total_Beloss = 0.0
    total_valid_acc = 0.0
    start_time = time.time()


    for batch, (arr, choice_frame, name, cls_labels, reg_labels) in enumerate(data_loader):
        model.eval()

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

        # print('Eval [current epoch: %2d total: %2d iter/all %4d/%4d]  Loss: %.4f \t time: %.3f @ %s' % (epoch, epoch_max_number, batch,
        #                                                                len(data_loader.dataset) / train_batch_size,
        #                                                                 running_loss.item(), time.time() - start_time,
        #                                                                     time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
        # print('batch total loss:  %4f  %4f  %4f ' % (cls_loss.item() ,reg_loss.item(), running_loss.item()))
        # print('%')

    valid_plot_arr_allloss.append([total_running_loss, total_closs, total_rloss, total_L1loss, total_L2loss, total_Beloss])

    np.savetxt('valid_plot_arr_allloss' + str(useHead) + '.csv', valid_plot_arr_allloss,
               delimiter=',', fmt='%.5f')

    print ("=============validition=====================")
    print('total allLoss:  %4f  total rloss:  %4f   total closs:  %4f ' % (total_running_loss, total_rloss, total_closs,))

    print('extr. prediction data vs label')
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    print(torch.mul(n, 100).squeeze(1).cpu().data.numpy())
    print(nl.float().data.numpy())

    print("\n\n")

def train_image():


    print("total valid simple : " + str(len(data_list)))



    if full_train:

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list, useHead=useHead,
                                                              modality_type=modality_type)
    else:

        train_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list[0:len(data_list) // 5 * 4], useHead=useHead,
                                                              modality_type=modality_type)
        valid_dataset = PersonalityDataset.PersonalityDataset('./dataset/train', 'train',
                                                              data_list[len(data_list) // 5 * 4: -1], useHead=useHead,
                                                              modality_type=modality_type)

        valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)



    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


    print("train simple:  " + str(len(train_dataset.data_list)))
#    print("dev simple:  " + str(len(valid_dataset.data_list)))



    for epoch in range(epoch_max_number):

        global learning_rate

        if useHead:
            model_name = './models/pf_pre_useHead_%d.model' % (epoch + 1)
        else:
            model_name = './models/pf_pre_useSence_%d.model' % (epoch + 1)

        train(data_loader=train_dataloader, epoch=epoch)
        if not full_train:
            valid(data_loader=valid_dataloader, epoch=epoch)


        if (epoch + 1) % 10 == 0:
            learning_rate = change_lr(optimizer,
                                      learning_rate)  # here the reference of learning_rate is also globally changed
        if (epoch + 1) % 5 == 0:
            torch.save(model, model_name)

    print_loss()

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
            #print(row)
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
            #print(row)
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

    print("train_feature_arr:       " + str(len(train_feature_vector)))
    print("train_feature_score:     " + str(len(train_label_vector)))
    print("test_feature_arr:        " + str(len(test_feature_vector)))
    print("test_feature_score:      " + str(len(test_label_vector)))

    result_list = []

    train_label_vector = np.array(train_label_vector)
    test_label_vector = np.array(test_label_vector)

    for i in range(5):

        ss_x = StandardScaler()

        x_train = ss_x.fit_transform(train_feature_vector)
        x_test = ss_x.transform(test_feature_vector)

        y_train = train_label_vector[:, i]
        y_test = test_label_vector[:, i]

        etr = ExtraTreesRegressor(n_estimators=1000,max_features=len(x_train[0]))
        etr.fit(x_train, y_train)

        etr_y_predict = etr.predict(x_test)

        result = 0

        for i in range(len(etr_y_predict)):
            result = result + abs(etr_y_predict[i] - y_test[i])

        result = result / len(etr_y_predict)

        print("extra tree eval: " + str(1 - result))

        result_list.append(1 - result)
    print(result_list)
    print(np.mean(result_list))

def train_audio():

    global learning_rate
    useIS_OS = False
    if useIS_OS :
        import scipy.io as scio
        path = './feature_extraction_result/OS_IS13.mat'

        data = scio.loadmat(path)

        # 读取数据 和训练集视频对应 6000个
        # print (data["OS_IS13"][0][0][0])

        dict_audio_data = {}

        for item in range(8000):
            file_name = data["OS_IS13"][0][0][0][item][0][0]
            # print(file_name)
            # print(data["OS_IS13"][0][0][1][item])
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
        print("total valid simple : " + str(len(data_list)))

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

    print_loss()

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

        # #print(e.size(), n.size())
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

        # print('Training [current epoch: %2d total: %2d iter/all %4d/%4d]  Loss: %.4f \t time: %.3f @ %s' % (epoch, epoch_max_number, batch,
        #                                                                len(data_loader.dataset) / train_batch_size,
        #                                                                 running_loss.item(), time.time() - start_time,
        #                                                                     time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
        # print('batch total loss:  %4f  %4f  %4f ' % (cls_loss.item() ,reg_loss.item(), running_loss.item()))
        # print('%')

    train_plot_arr_allloss.append([total_rloss, total_closs, total_rloss, total_L1loss, total_L2loss, total_Beloss])

    if (epoch + 1) % 10 == 0:
        np.savetxt('train_plot_arr_allloss' + str(useHead) + '.csv', train_plot_arr_allloss,
                   delimiter=',', fmt='%.5f')

    print("epoch: " + str(epoch + 1))
    print("=============training=====================")
    print('total allLoss:  %4f  total rloss:  %4f   total closs:  %4f ' % (
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

    if modality_type == 2:
        train_audio()
    else:
        train_image()

    print(div_arr, modality_type, resnet_pretrained)

    #print_loss()

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
    # print(tmp1, tmp2, tmp3, tmp4)

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
    # print(len(file_list_test))
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
    #     print(tmp)