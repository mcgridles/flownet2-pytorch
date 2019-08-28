import os
import argparse
import numpy as np
import cv2
import colorama
from imageio import imread

import torch
import torch.nn as nn
from torch.autograd import Variable

import models
import losses
from utils import frame_utils, tools


class ModelAndLoss(nn.Module):
    def __init__(self, args):
        super(ModelAndLoss, self).__init__()
        kwargs = tools.kwargs_from_args(args, 'model')
        self.model = args.model_class(args, **kwargs)
        kwargs = tools.kwargs_from_args(args, 'loss')
        self.loss = args.loss_class(args, **kwargs)

    def forward(self, data, target, inference=True):
        output = self.model(data)

        loss_values = self.loss(output, target)

        if not inference:
            return loss_values
        else:
            return loss_values, output


def preprocess(args):
    '''
    Prepare images for inference and convert to torch tensors.
    '''

    class StaticCenterCrop(object):
        def __init__(self, image_size, crop_size):
            self.th, self.tw = crop_size
            self.h, self.w = image_size
        def __call__(self, img):
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

    frame_size = frame_utils.read_gen(args.images[0]).shape
    render_size = args.inference_size
    if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
        render_size[0] = ((frame_size[0]) // 64) * 64
        render_size[1] = ((frame_size[1]) // 64) * 64

    img1 = imread(args.images[0])
    img2 = imread(args.images[1])
    images = [img1, img2]
    image_size = img1.shape[:2]

    cropper = StaticCenterCrop(image_size, render_size)
    images = list(map(cropper, images))

    images = np.array(images).transpose(3, 0, 1, 2)
    images = torch.from_numpy(images.astype(np.float32))
    target = torch.zeros((2,) + images.size()[-2:])

    return [images], [target]


def inference(data, target, model):
    '''
    Perform inference (calculate optical flow) for two images.
    '''

    model.eval()

    # Expand dimension for batch size
    for i in range(len(data)):
        data[i] = data[i].unsqueeze(0)
        target[i] = target[i].unsqueeze(0)

    data = [Variable(d) for d in data]
    target = [Variable(t) for t in target]

    with torch.no_grad():
        losses, output = model(data[0], target[0])

    losses = [torch.mean(loss_value) for loss_value in losses]

    return losses, output


def parse_args():
    '''
    Parse and prepare command line arguments.
    '''

    parser = argparse.ArgumentParser()

    # CUDA args
    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    # FlowNet args
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--rgb_max", type=float, default=255.0)
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

    # Model and loss args
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    # Custom args
    parser.add_argument('--weights', '-wt', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--images', '-im', nargs=2, type=str)

    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():
    '''
    Command for running on capstone4790-vm-1 (IP: 35.197.106.62):
    >>> python capstone_pipeline.py --weights /mnt/disks/datastorage/weights/FlowNet2_checkpoint.pth.tar --images /mnt/disks/datastorage/MPI-Sintel/training/final/alley_1/frame_0001.png /mnt/disks/datastorage/MPI-Sintel/training/final/alley_1/frame_0002.png --model FlowNet2
    '''

    args = parse_args()

    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        model_and_loss = ModelAndLoss(args)

        block.log('Number of parameters: {}'.format(
            sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # Passing to cuda or wrap with data parallel, model and loss
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed)
        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()

            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)
        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        if os.path.isfile(args.weights):
            block.log("Loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.weights, checkpoint['epoch']))
        else:
            block.log("No checkpoint found at '{}'".format(args.weights))
            quit()

    with tools.TimerBlock("Performing inference") as block:
        block.log("Preprocessing")
        images, target = preprocess(args)

        block.log('Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in images])))
        block.log('Inference Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in target])))

        block.log("Inference")
        losses, output = inference(data=images, target=target, model=model_and_loss)

        output = output.cpu()       # convert to CPU tensor
        output = output.squeeze(0)  # remove batch dimension
        output = output.numpy()     # convert to numpy array
        
        block.log(losses)
        block.log(output.shape)


if __name__ == '__main__':
    main()

