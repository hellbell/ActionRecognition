import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

import time
import numpy as np
from PIL import Image
from vgg import Vgg16
from model import ActionClassifier
from database.database import Dataset
import os

# Initialize Networks
vgg = Vgg16(requires_grad=False)
vgg = vgg.cuda()
classifier = ActionClassifier()
classifier = classifier.cuda()
save_path = os.path.join('nets', 'epoch_%d.pth' % 450)
classifier.load_state_dict(torch.load(save_path))


# Load dataset
dataset = Dataset()
dataset.initialize()
dataset.set_test_mode()

num_test_vids = len(dataset)
batch_size = 64
input_size = 224

# pred_labels = np.zeros((num_test_vids, ))
# gt_labels =  np.zeros((num_test_vids, ))

pred_labels_all = []
gt_labels_all = []

transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

correct_count = 0

for vid_idx in range(num_test_vids):
    img_list, labels = dataset.get_test_data(vid_idx)

    num_imgs = len(img_list)
    num_batches = num_imgs // batch_size
    input_images = Variable(
        torch.cuda.FloatTensor(
            batch_size, 3, input_size, input_size))

    pred_labels = np.zeros((num_imgs, ))
    gt_labels = np.zeros((num_imgs, ))

    for batch_idx in range(num_batches):
        count = 0
        batch_idx_offset = batch_idx * batch_size
        batch_idx_end = min(batch_idx_offset + batch_size, num_imgs)
        curr_idxs = np.arange(batch_idx_offset, batch_idx_end)

        for i in range(batch_idx_offset, batch_idx_end):
            img = Image.open(img_list[i]).convert('RGB')
            img = Variable(transform(img))
            input_images[count] = img.cuda()
            count = count + 1

        vgg_out = vgg(input_images)
        cls_out = classifier(vgg_out)
        cls_out = cls_out.cpu().data.numpy()
        cls_out_max = np.argmax(cls_out, 1)

        pred_labels[curr_idxs] = cls_out_max[:len(curr_idxs)]
        gt_labels[curr_idxs] = np.asarray(labels)

    pred_labels_all.append(pred_labels)
    gt_labels_all.append(gt_labels)

    pred_labels_counts = np.bincount(np.int64(pred_labels))
    pred_action_cls = np.argmax(pred_labels_counts)
    if (labels == pred_action_cls):
        correct_count = correct_count + 1
        print('correct!')
    else:
        print('incorrect!')

    corrects = np.sum(gt_labels == pred_labels)
    corrects_ratio = float(corrects) / float(num_imgs)
    # print('label: %d, accuracy: %f' %(labels, corrects_ratio))


print ('Action classification accuracy: %f' % (float(correct_count)/float(num_test_vids)))

