import os.path
import random
import torch
from PIL import Image
import numpy as np
from .image_folder import make_dataset
from torchvision import transforms
from torch.autograd import Variable


class Dataset():
    def __init__(self):
        self.data = []
        self.labels = []
        self.cls_names = []
        self.train_idx = []
        self.test_idx = []
        self.input_size = 224
        self.batch_size = 64
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        self.is_train = True

    def initialize(self):
        data_root = '/home/sangdoo/work/dataset/UCF11'
        root_dir = os.listdir(data_root)
        root_dir = sorted(root_dir)

        self.classes = root_dir

        np.random.seed(1000)



        for i in range(len(root_dir)):
            cls_name = root_dir[i]
            vid_path = os.path.join(data_root, cls_name)
            vid_dir  = os.listdir(vid_path)
            vid_dir  = sorted(vid_dir)

            num_vids = len(vid_dir)
            data_idx_offset = len(self.data)
            for j in range(len(vid_dir)):
                vid_name = vid_dir[j]
                video_path = os.path.join(vid_path, vid_name)
                imgs = sorted(make_dataset(video_path))
                self.data.append(imgs)
                self.labels.append(i)
                self.cls_names.append(cls_name)


            all_idx = data_idx_offset + np.random.permutation(num_vids)
            self.train_idx = np.append(self.train_idx, all_idx[:int(num_vids * 0.7)])
            self.test_idx  = np.append(self.test_idx,  all_idx[int(num_vids * 0.7):])

        self.train_idx = np.int64(self.train_idx)
        self.test_idx = np.int64(self.test_idx)

            # self.train_idx.append(all_idx[:int(num_vids * 0.7)])
            # self.test_idx.append(all_idx[int(num_vids * 0.7):])




        # self.all_idx = np.random.permutation(len(self.data))
        # self.train_idx = self.all_idx[:int(len(self.data) * 0.7)]
        # self.test_idx  = self.all_idx[int(len(self.data) * 0.7):]

        # self.train_data = self.data[train_idx]
        # self.test_data = self.data[test_idx]
        # self.train_labels = self.labels[train_idx]
        # self.test_labels = self.labels[test_idx]

    def get_test_data(self, vid_idx):
        img_list = self.data[self.test_idx[vid_idx]]
        label = self.labels[self.test_idx[vid_idx]]
        return img_list, label



    def __getitem__(self, index):

        labels_ = []
        data_ = []

        input_images = Variable(torch.cuda.FloatTensor(self.batch_size, 3, self.input_size, self.input_size))

        count = 0
        for i in index:
            if self.is_train:
                img_list = self.data[self.train_idx[i]]
                label = self.labels[self.train_idx[i]]
            else:
                img_list = self.data[self.test_idx[i]]
                label = self.labels[self.test_idx[i]]

            r = np.random.randint(0, len(img_list))
            data_.append(img_list[r])
            labels_.append(label)

            img = Image.open(img_list[r]).convert('RGB')
            img = Variable(self.transform(img))
            input_images[count] = img.cuda()
            count = count + 1

        return input_images, labels_



    def __len__(self):
        if self.is_train:
            return (len(self.train_idx))
        else:
            return (len(self.test_idx))

    def set_train_mode(self):
        self.is_train = True

    def set_test_mode(self):
        self.is_train = False


    def name(self):
        return 'TrainDataset'
