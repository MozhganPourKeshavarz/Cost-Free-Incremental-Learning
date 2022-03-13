# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel


import torch
import torchvision
import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
import torch.utils.data as data_utl
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data_utl
from torchvision import transforms
from PIL import Image

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        # print(loss)

        if not self.buffer.is_empty():
            # print("read from the buffer ...")
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size)#, transform=self.transform
            buf_outputs = self.net(buf_inputs)
            # print(buf_logits.shape)
            # print(buf_outputs.shape)

            # print(buf_logits[6])
            # print(buf_outputs[6])
            # print(F.mse_loss(buf_outputs, buf_logits))
            # print(self.args.alpha)
            # exit()
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            # print(loss)
            # exit()

        loss.backward()
        self.opt.step()
        # self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()

    # def update_buffer(self, data_impressions, data_impressions_logits):
    #     print(type(data_impressions))
    #     self.buffer.add_data(examples=data_impressions, logits=data_impressions_logits.data)

    def update_buffer_lenet5(self, data_impressions, data_impressions_logits):
        # print(type(data_impressions))
        self.buffer.add_data(examples=data_impressions, logits=data_impressions_logits.data)


    def update_buffer_cifar10(self, data_impressions, data_impressions_logits):
        # print(type(data_impressions))
        for img , logits in zip(data_impressions, data_impressions_logits):
            self.buffer.add_data(examples=img, logits=logits.data)

