import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule


def test(args):
    # output folder
    outdir = '%s/%s'%(args.dataset, args.prediction_dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform)
    else:#set split='test' for test set
        testset = get_segmentation_dataset(args.dataset, split='test', mode='vis',
                                           transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, norm_layer=BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        print("===>> loading model file: ",args.resume)
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    print(model)
    num_class = testset.num_class
    print("testing mode [ss(0), ms(1)]: ",args.multi_scales)
    evaluator = MultiEvalModule(model, testset.num_class, multi_scales=args.multi_scales).cuda()
    evaluator.eval()

    tbar = tqdm(test_data)
    def eval_batch(image, dst, evaluator, eval_mode):
        if eval_mode:
            # evaluation mode on validation set
            targets = dst
            #print("image.len: ", len(image), "image[0].size: ", image[0].size())
            outputs = evaluator.parallel_forward(image)
            
            batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
            for output, target in zip(outputs, targets):
                print("output.size: ", output.size(), "target.size: ",target.size())
                correct, labeled = utils.batch_pix_accuracy(output.data.cpu(), target)
                inter, union = utils.batch_intersection_union(
                    output.data.cpu(), target, testset.num_class)
                batch_correct += correct
                batch_label += labeled
                batch_inter += inter
                batch_union += union
            return batch_correct, batch_label, batch_inter, batch_union
        else:
            # Visualize and dump the results
            #print("image.len: ", len(image), "image[0].size: ", image[0].size())
            im_paths = dst
            outputs = evaluator.parallel_forward(image)
            predicts = [torch.max(output, 1)[1].cpu().numpy() + testset.pred_offset
                        for output in outputs]
            for predict, impath in zip(predicts, im_paths):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))
            # dummy outputs for compatible with eval mode
            return 0, 0, 0, 0

    total_inter, total_union, total_correct, total_label = \
        np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    for i, (image, dst) in enumerate(tbar):
        with torch.no_grad():
            correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval)
        if args.eval:
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
            
            return pixAcc, mIoU, IoU, num_class

def eval_multi_models(args):
    if args.resume_dir is None or not os.path.isdir(args.resume_dir):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))
    #resume_file_list = os.listdir(args.resume_dir)
    resume_file_list =[]
    resume_file_list.append(args.resume_file)
    for resume_file in resume_file_list:
        print("resume file:", resume_file)
        if os.path.splitext(resume_file)[1] == '.tar':
            args.resume = os.path.join(args.resume_dir, resume_file)
            print("resume file path:", args.resume)
            assert os.path.exists(args.resume)

            if args.eval: # for val set
                pixAcc, mIoU, IoU, num_class = test(args)
        
                txtfile = args.resume
                txtfile = txtfile.replace('pth.tar', 'txt')
                if not args.multi_scales:
                    txtfile = txtfile.replace('.txt', 'result_mIoU_%.4f.txt'%mIoU)
                else:
                    txtfile = txtfile.replace('.txt', 'multi_scale_result_mIoU_%.4f.txt'%mIoU)
                fh = open(txtfile, 'w')
                print("================ Summary IOU ================\n")
                for i in range(0,num_class):
                    print("%3d: %.4f\n" %(i,IoU[i]))
                    fh.write("%3d: %.4f\n" %(i,IoU[i]))
                print("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
                print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
                fh.write("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
                fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
                fh.close()
            else : # for test set
                test(args)

    print('Evaluation is finished!!!')



def get_arguments():
    parser = argparse.ArgumentParser(description='Segmentation, TKCN_PyTorch')
    # model and dataset 
    parser.add_argument('--model', type=str, default='tkcnet', help='model name (default: tkcnet, fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101', help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset name (default: cityscapes')
    parser.add_argument('--data-folder', type=str,  default="/home/wty/AllDataSet/", help='training dataset folder')
    parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=2048, help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024, help='crop image size')

    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N', help='input batch size for \
                        testing (default: same as batch size)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume-dir', type=str, default='cityscapes/model/tkcnet_model_resnet101_cityscapes_gpu6bs6epochs240/TKCNet101',
                        help='put the path to resuming dir if needed')
    parser.add_argument('--resume-file', type=str, default='checkpoint_240.pth.tar', help='model file')
    #parser.add_argument('--resume-file', type=str, default='model_best.pth.tar', help='model file')
    parser.add_argument('--checkname', type=str, default='default', help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None, help='evaluating on model zoo model')

    # evaluation option
    parser.add_argument('--ema', action='store_true', default= False, help='using EMA evaluation')
    parser.add_argument('--eval', action='store_true', default= False, help='evaluating mIoU, true for val set; false for test set')
    
    parser.add_argument('--multi-scales',action="store_true", default=False, help="for testing, false: ss, true: ms)")
    parser.add_argument('--prediction_dir', type=str, default='seg_predictions', help='the prediction results of model')
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    print(args)
    eval_multi_models(args)
