import argparse

from torchvision import models

from utils import *
from MRP import MRP
from trainer.learner_train import t_trainer


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10', help='[mnist, cifar10, cifar100]')
    parser.add_argument('--t_train', type=str2bool, default='False', help='Train learner network?')
    parser.add_argument('--num_sample', type=int, default=24000, help='Number of samples crafted per category')
    parser.add_argument('--beta', type=list, default=[0.1, 1.], help='Beta  scaling vectors')
    parser.add_argument('--t', type=int, default=20, help='Temperature for distillation')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--iters', type=int, default=1500, help='iteration number')
    parser.add_argument('--s_save_path', type=str, default='./saved_model/', help='save path for student network')

    args = parser.parse_args()
    t_model_path='./trainer/models/learner_'+args.dataset+'.pt'

    if args.t_train == True:
        T_trainer=t_trainer(args.dataset)
        T_trainer.build()
    teacher = load_model(args.dataset, t_model_path)

    mrp = MRP(args.dataset, teacher, args.num_sample, args.beta, args.t, args.batch_size, args.lr, args.iters)
    _ , save_root = mrp.build()




if __name__ == "__main__":
    set_gpu_device(0)
    main()