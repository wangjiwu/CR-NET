from torchvision import transforms, utils
from torch.utils.data import Dataset
import random
import math
# import librosa
#import librosa
import cv2
import os
import numpy as np
import torch
from config import div_arr,random_seclect

import six.moves.cPickle as pickle



def get_data_list_and_big_five(ds_file_root, typ):
    # logger.info('dataset file is ', ds_file_root)

    with open('./dataset/%s/annotation_%s.pkl' % (typ, typ), 'rb') as fo:  # 读取pkl文件数据
        r = pickle.load(fo, encoding='bytes')

    # r = pickle.load(open('./dataset/%s/annotation_%s.pkl' %(typ, typ), encoding='bytes')) #  encoding='bytes' is only for python 3
    dataset = []

    for root, dirs, files in os.walk(ds_file_root + "/new_extracted_image"):
    #for root, dirs, files in os.walk(ds_file_root + "/extracted_image"):
        files_list = []

        for item in files:
            files_list.append(item[:-4] + ".mp4")
        try:
            files_list.remove(".mp4")
        except:
            pass
        files = files_list

        dataset = zip(files, ((lambda x: r[b'extraversion'][x])(f.encode(encoding="utf-8")) for f in files),
                      ((lambda x: r[b'neuroticism'][x])(f.encode(encoding="utf-8")) for f in files),
                      ((lambda x: r[b'agreeableness'][x])(f.encode(encoding="utf-8")) for f in files),
                      ((lambda x: r[b'conscientiousness'][x])(f.encode(encoding="utf-8")) for f in files),
                      ((lambda x: r[b'openness'][x])(f.encode(encoding="utf-8")) for f in files),
                      ((lambda x: r[b'interview'][x])(f.encode(encoding="utf-8")) for f in files)
                      )
        dataset = list(dataset)
        # the mapping is video_name, E_value, N_value, A_value, C_value, O_value, I_value

    return dataset


def get_one_hot(array_s, devide_arr= div_arr):
    arr = []

    for item in array_s:

        for i in range(len(devide_arr)):
            if item < devide_arr[i]:
                arr.append(i)
                break;
            else:
                continue

        if item >= devide_arr[-1]:
            arr.append(len(devide_arr))

    return arr

class PersonalityDataset(Dataset):

    def __init__(self, path, typ, data_list, useHead, modality_type = 2):
        fr = open('./dataset/trainvec.pkl', 'rb')
        self.text_train = pickle.load(fr,encoding='iso-8859-1')
        fr = open('./dataset/testvec.pkl', 'rb')
        self.text_test = pickle.load(fr,encoding='iso-8859-1')

        self.modality_type = modality_type
        self.data_list = data_list
        self.dataset_path = path
        self.useHead = useHead
        self.typ = typ



    def __getitem__(self, index):

        if self.modality_type == 0 or  self.modality_type == 1:
            if not  random_seclect:
                #data_path = self.dataset_path + '/videos_face/' + self.data_list[index][0]
                filename =  self.data_list[index][0][:-4] + ".pkl"
                data_path = self.dataset_path + '/new_extracted_image/' + filename
                #data_path = self.dataset_path + '/extracted_image/' + filename

                #data_path = self.dataset_path + '/preTrain_feature/' + filename

                #  index indicates the order in the list and [0] means its address, see the tuple example above.
                arr = []

                pkl_file = open(data_path, 'rb')

                data = pickle.load(pkl_file)
                sence_series = data["sence_series"]
                head_series =  data["head_series"]


                if not self.useHead:

                    arr = [
                        transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).view(3, 112, 112, 1)
                        for frame in sence_series]
                else:
                    arr = [
                        transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).view(3, 112, 112, 1)
                        for frame in head_series]
                arr = torch.cat(arr, dim=3)

                # 4-cls


                cls_label = get_one_hot(self.data_list[index][1:])

            else:
                # data_path = self.dataset_path + '/videos/' + self.data_list[index][0]
                # #  index indicates the order in the list and [0] means its address, see the tuple example above.
                # arr = []
                # frame_num = 32
                # cap = cv2.VideoCapture(data_path)
                # while cap.isOpened():
                #     ret, frame = cap.read()
                #     if ret == True:
                #         arr.append(frame)
                #     else:
                #         break
                # cap.release()
                #
                #
                # if self.typ == 'train':
                #     f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / frame_num,
                #                          range(int(n * i / frame_num), int(n * (i + 1) / frame_num)))
                #                                                                     for i in range(frame_num)]
                #     sl = f(len(arr))
                #
                #
                #
                #
                # else:
                #     sl = [int(i*math.floor(len(arr)/frame_num)) for i in range(frame_num)]
                #
                # arr = [arr[i] for i in sl]

                #--------------------------使用提取好的112*112

                # data_path = self.dataset_path + '/videos_face/' + self.data_list[index][0]
                filename = self.data_list[index][0][:-4] + ".pkl"
                if self.typ == 'train':
                    data_path = self.dataset_path + '/total_new_extracted/' + filename
                else:
                    data_path = self.dataset_path + '/new_extracted_image/' + filename


                frame_num = 32

                #  index indicates the order in the list and [0] means its address, see the tuple example above.
                arr = []

                pkl_file = open(data_path, 'rb')

                data = pickle.load(pkl_file)
                sence_series = data["sence_series"]
                head_series = data["head_series"]



                if self.typ == 'train':
                    f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / frame_num,
                                         range(int(n * i / frame_num), int(n * (i + 1) / frame_num)))
                                                                                    for i in range(frame_num)]
                    sl = f(len(sence_series))




                else:
                    sl = [int(i) for i in range(frame_num)]




                if not self.useHead:


                    arr_sence = [sence_series[i] for i in sl]

                    arr = [
                        transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).view(3,112,112,1)
                        for frame in arr_sence]

                else:
                    arr_head = [head_series[i] for i in sl]

                    arr = [
                        transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).view(3,112,112,1)
                        for frame in arr_head]
                arr = torch.cat(arr, dim=3)



                cls_label = get_one_hot(self.data_list[index][1:])




            return arr , [filename], self.data_list[index][0], cls_label, [x*100 for x in self.data_list[index][1:]] #, self.mean_five

        else:



            # data_path = self.dataset_path + '/videos_face/' + self.data_list[index][0]
            filename = self.dataset_path + '/audio/' + self.data_list[index][0][:-4] + ".npy"

            y = np.load(filename)

            y_downsample = []

            for i in range (y.shape[0]):
                if (i % 4 == 0):
                    y_downsample.append(y[i])

            y_downsample = np.array(y_downsample)

            cls_label = get_one_hot(self.data_list[index][1:])

            # load text vector

            if self.typ == "train":
                text_vec = self.text_train[(self.data_list[index][0][:-4] + ".mp4")]
            else:
                try:
                    text_vec = self.text_test[(self.data_list[index][0][:-4] + ".mp4")]
                except:
                    text_vec = np.array([0.0]*4800 )


            text_vec = np.array(text_vec,dtype=np.float32)

            y_downsample = np.concatenate((y_downsample, text_vec), axis=0)


            return y_downsample, [], self.data_list[index][0], cls_label, [x * 100 for x in
                                                                  self.data_list[index][1:]]  # , self.mean_five



    def __len__(self):
        return len(self.data_list)
