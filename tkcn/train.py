import os
import copy
import torch
import timeit
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.models import get_segmentation_model
from encoding.nn import SegmentationLosses,BatchNorm2d
from encoding.datasets import get_segmentation_dataset
from encoding.nn import SegmentationMultiLosses, SegmentationSingleLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion


class Trainer():
    def __init__(self, args):
        self.args = args
        args.log_name = str(args.checkname)
        args.log_root = os.path.join(args.dataset, args.log_root) # dataset/log/
        self.logger = utils.create_logger(args.log_root, args.log_name)
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size, 'logger': self.logger,
                       'scale': args.scale}
        trainset = get_segmentation_dataset(args.dataset, split='trainval', mode='trainval',
                                            **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode='val',  # crop fixed size as model input
                                           **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone,
                                       norm_layer=BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       )
        #print(model)
        self.logger.info(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            print("this model has object, head")
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        self.criterion = SegmentationLosses(nclass=self.nclass)
        
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        
        # resuming checkpoint
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader), logger=self.logger,
                                            lr_step=args.lr_step)
        self.best_pred = 0.0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)

        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        self.logger.info('Train loss: %.3f' % (train_loss / (i + 1)))

        if not self.args.no_val:
            # save checkpoint every 10 epoch
            filename = "checkpoint_%s.pth.tar"%(epoch+1)
            is_best = False
            if  epoch > self.args.epochs-10:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                    }, self.args, is_best, filename)
            elif not epoch % 10:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                    }, self.args, is_best, filename)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')

        for i, (image, target) in enumerate(tbar):
            with torch.no_grad():
                correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
        self.logger.info('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)

def get_arguments():
    parser = argparse.ArgumentParser(description='Segmentation: TKCN_PyTorch')
    
    # model and dataset 
    parser.add_argument('--model', type=str, default='tkcnet', help='model name (default: tkcnet, or fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101', help='backbone name (default: resnet50, resnet101)')
    parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset name (default: cityscapes, voc12, pcontext)')
    parser.add_argument('--data-folder', type=str, default="/home/wty/AllDataSet/", help='training dataset folder')
    parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default= 1024, help= 'base image size: 1024 for cityscapes')
    parser.add_argument('--crop-size', type=int, default= 768, help= 'crop image size: 768 for cityscapes')
    
    # training hyper params
    parser.add_argument('--epochs', type=int, default= 240, metavar='N', help='240 for cityscapes')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N', 
                           help='input batch size for testing (default: same as batch size)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: auto, 0.003 for Cityscapes)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='learning rate scheduler (default: poly)')
    parser.add_argument('--lr-step', type=int, default=None, help='lr step to change lr')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 1e-4)')
    
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default= False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-root', type=str, default='log', help='set a log path folder')

    # checking point
    parser.add_argument('--resume', type=str, default='',
                        #'./cityscapes/model/tkcnet_model_resnet101_cityscapes_gpu6bs6epochs240/TKCNet101/checkpoint_61.pth.tar',
                        help='put the path to resuming file if needed')
    
    parser.add_argument('--resume-dir', type=str, default=None, help='put the path to resuming dir if needed') #
    parser.add_argument('--checkname', type=str, default='TKCNet101',help='set the checkpoint name, default: TKCNet101, fcn_resnet50')
    parser.add_argument('--model-zoo', type=str, default=None, help='evaluating on model zoo model')
    parser.add_argument('--ft', action='store_true', default= False, help='finetuning on a different dataset')
    
    # finetuning pre-trained models
    parser.add_argument('--pre-class', type=int, default=None, help='num of pre-trained classes (default: None)')

    # evaluation option
    parser.add_argument('--ema', action='store_true', default= False, help='using EMA evaluation')
    parser.add_argument('--eval', action='store_true', default= False, help='evaluating mIoU')
    parser.add_argument('--no-val', action='store_true', default= False, help='skip validation during training')

    # test option
    parser.add_argument('--test-folder', type=str, default=None, help='path to test image folder')
    parser.add_argument('--multi-scales',action="store_true", default=False, help="for testing,  False: ss, Ture: ms ")
    parser.add_argument('--scale', action='store_false', default=True, 
                         help='for training, choose to use random scale transform(0.75-2),default:multi scale')
    # GPU configuration
    parser.add_argument("--cuda", default='', help="Run on CPU or GPU")
    parser.add_argument("--gpu_nums", type=int, default=0, help="the number of gpus.")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N', help='input batch size for training (default: auto)')
    # the parser
    return parser.parse_args()


if __name__ == "__main__":
    start = timeit.default_timer()
    args = get_arguments()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.batch_size is None:
        args.batch_size = torch.cuda.device_count()
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    args.gpu_nums = torch.cuda.device_count()
    
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.logger.info(['Starting Epoch:', str(args.start_epoch)])
    trainer.logger.info(['Total Epoches:', str(args.epochs)])

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation(epoch)
    end = timeit.default_timer()
    print("training time:", 1.0*(end-start)/3600)
    trainer.logger.info('training time: %.3f  h'%(1.0*(end - start)/3600))
