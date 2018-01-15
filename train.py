import torch
from torch.autograd import Variable
import torch.nn.functional as F

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


# Load dataset
dataset = Dataset()
dataset.initialize()

num_epoch = 500
num_train_data = len(dataset)
batch_size = 64

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))

tt = time.time()

for epoch in range(num_epoch):
    curr_epoch_idx = np.random.permutation(num_train_data)
    num_batches = num_train_data // batch_size
    loss_sum = 0
    for batch_idx in range(num_batches):
        batch_idx_offset = batch_idx * batch_size
        batch_train_idx = curr_epoch_idx[batch_idx_offset:batch_idx_offset + batch_size]

        # print (batch_idx)
        input_images, labels = dataset[batch_train_idx]

        optimizer.zero_grad()

        vgg_out = vgg(input_images)
        cls_out = classifier(vgg_out)

        labels = np.asarray(labels)
        labels_tensor = Variable(torch.from_numpy(labels))
        labels_tensor = labels_tensor.cuda()

        loss = F.nll_loss(cls_out, labels_tensor)

        loss.backward()
        optimizer.step()

        loss_sum = loss_sum + loss


    if (epoch % 50 == 0):
        save_path = os.path.join('nets', 'epoch_%d.pth' % epoch)
        torch.save(classifier.cpu().state_dict(), save_path)
        classifier.cuda()


    ## testing
    dataset.set_test_mode()
    num_test_data = len(dataset)
    test_idx = np.random.permutation(num_test_data)
    input_images, labels = dataset[test_idx[:batch_size]]
    labels = np.asarray(labels)
    labels_tensor = Variable(torch.from_numpy(labels))
    labels_tensor = labels_tensor.cuda()

    vgg_out = vgg(input_images)
    cls_out = classifier(vgg_out)
    test_loss = F.nll_loss(cls_out, labels_tensor)

    print('epoch: %03d, train_loss: %f, test_loss: %f, %f sec' % (epoch, loss_sum/num_batches, test_loss, time.time() - tt))

    dataset.set_train_mode()
