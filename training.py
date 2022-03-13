# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
from PIL import Image
import scipy.io as sio
import cv2, os

def data_impression_Lenet5(model: ContinualModel,finished_calss):
    # %% Visualize generated DI samples
    data_result = []
    count = 0
    home_path = "/Users/mozhgan/Documents/PycharmProjects/CL_DER"
    data = sio.loadmat(home_path + '/data/data_imp/Orginal__DI-1.mat')
    for i, l in zip(data['train_images'], data['train_labels']):
        first, second = sorted(np.vstack([np.arange(10), l]).T, key=lambda l: l[1], reverse=True)[:2]

        if int(first[0]) in finished_calss:
            data_result.append([int(first[0]), first[1], i])

            try:
              os.mkdir(home_path+"/data/data_imp/"+str(first[0]))
            except:
                pass
                # print("already exists")

            image = np.pad(cv2.resize(i, (200, 200)).astype(np.uint8), [[24, 0], [0, 0]], 'constant')
            cv2.putText(image, '\'%d\':%.2f, \'%d\':%.2f' % (first[0], first[1], second[0], second[1]),
                        (0, 24), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imwrite(home_path +"/data/data_imp/" + str(first[0]) + "/"+ str(count) +".jpg", image)
            count += 1

    data_result.sort(key=lambda x: x[1])
    data_impressions = data_result[-50:]

    data_impressions = np.expand_dims(np.array([cv2.resize(ti[2], (28, 28)) for ti in data_impressions]), -1)
    data_impressions = [di.transpose([2, 0, 1]) for di in data_impressions]
    data_impressions  = np.asarray(data_impressions)
    data_impressions = torch.from_numpy(data_impressions)

    # print(type(data_impressions))
    # print(data_impressions.shape)

    # print("")
    # print("Data Impression ... ")
    status = model.net.training
    model.net.eval()

    data_impressions_logits = model(data_impressions)

    model.net.train(status)
    return data_impressions, data_impressions_logits

def data_impression_CIFAR10(model: ContinualModel,finished_calss):
    # %% Visualize generated DI samples
    data_result = []
    count = 0
    home_path = "/Users/mozhgan/Documents/PycharmProjects/CL_DER"

    images = np.load('./dl/X_T20_40000_lr_0.001_batch100_1500_iterations.npy')
    labels = np.load('./dl/ySoft_T20_40000_lr_0.001_batch100_1500_iterations.npy')


    for i, l in zip(images, labels):
        first, second = sorted(np.vstack([np.arange(10), l]).T, key=lambda l: l[1], reverse=True)[:2]

        if int(first[0]) in finished_calss:
            data_result.append([int(first[0]), first[1], i])

            try:
              os.mkdir(home_path+"/data/data_imp/cifar10/"+str(first[0]))
            except:
                pass
                # print("already exists")

            image = np.pad(cv2.resize(i, (100, 100)).astype(np.uint8), [[24, 0], [0, 0]], 'constant')
            cv2.putText(image, '\'%d\':%.2f, \'%d\':%.2f' % (first[0], first[1], second[0], second[1]),
                        (0, 24), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imwrite(home_path +"/data/data_imp/cifar10/" + str(first[0]) + "/"+ str(count) +".jpg", image)
            count += 1

    data_result.sort(key=lambda x: x[1])
    data_impressions = data_result[-50:]

    data_impressions = np.expand_dims(np.array([cv2.resize(ti[2], (28, 28)) for ti in data_impressions]), -1)
    data_impressions = [di.transpose([2, 0, 1]) for di in data_impressions]
    data_impressions  = np.asarray(data_impressions)
    data_impressions = torch.from_numpy(data_impressions)

    # print(type(data_impressions))
    # print(data_impressions.shape)

    # print("")
    # print("Data Impression ... ")
    status = model.net.training
    model.net.eval()

    data_impressions_logits = model(data_impressions)

    model.net.train(status)
    return data_impressions, data_impressions_logits



def data_imp(model: ContinualModel, finished_calss):
    print("")
    print("Data Impression ... ")
    home_path = "/Users/mozhgan/Documents/PycharmProjects/CL_DER"
    try:
        os.mkdir(home_path + "/data/data_impressions_by_me_lenet5/" + str(finished_calss))
    except:
        pass
    status = model.net.training
    model.net.eval()
    # optimizer = torch.optim.SGD(params=[input], lr=0.01)

    for di in range(5):
        input = torch.nn.parameter.Parameter(torch.empty(1, 1, 28, 28).uniform_(0, 1))
        optimizer = torch.optim.Adam(params=[input], lr=0.01)
        optimizer.zero_grad()
        for step in range(5000):
            i = 0  # np.random.randint(36-32)
            j = 0  # np.random.randint(36-32)
            X = input[:, :, i:i + 28, j:j + 28]
            # X = X * np.random.uniform(low=0.9, high=1.1)
            logits = model(X.float())  # .cuda()
            # loss=criterion(logits,target)
            loss = -logits[0][finished_calss]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print('step {} - loss is : {}'.format(step,loss.item()))
            # lr_sched.step()
            if step % 100 == 0 or step % 100 == 0:
                a = input.detach().numpy().squeeze()  # .transpose([1, 2, 0])
                b = a - np.min(a)
                b = 255 * b / np.max(b)
                temp_img = b.astype(np.uint8)
                pil_img = Image.fromarray(temp_img)  # .resize(size=(64, 64))
                pil_img.save(home_path+"/data/data_impressions_by_me_lenet5/" + str(
                    finished_calss) + "/" + str(di) + "_" + str(step) + ".jpg")
        # print(type(input))
        input.requires_grad = False
        if di == 0:
            data_impressions = input
        else:
            data_impressions = torch.cat((data_impressions, input), 0)


    print(data_impressions.shape)
    data_impressions_logits = model(data_impressions)

    model.net.train(status)
    return data_impressions, data_impressions_logits

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0


        finished_calss = np.unique(labels.numpy())
        # print()
        # print(finished_calss)
        # data_impressions, data_impressions_logits  = data_imp(model, finished_calss)
        # model.update_buffer(data_impressions, data_impressions_logits)
        data_impressions, data_impressions_logits = data_impression_CIFAR10(model,finished_calss)
        #
        # print(data_impressions.shape)
        # print(data_impressions_logits.shape)
        model.update_buffer_cifar10(data_impressions, data_impressions_logits)



        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
