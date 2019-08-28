import os
import argparse
import numpy as np
import cv2
from scipy.misc import imread

import torch
import torch.nn as nn

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
    render_size = args.inference_size

    frame_size = frame_utils.read_gen(args.images[0]).shape

    if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
        render_size[0] = ((frame_size[0]) // 64) * 64
        render_size[1] = ((frame_size[1]) // 64) * 64

    img1 = imread(args.images[0])
    img2 = imread(args.images[1])
    images = [img1, img2]

    images = np.array(images).transpose(3, 0, 1, 2)
    images = torch.from_numpy(images.astype(np.float32))

    target = torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])

    return images, target


def inference(data, target, model):
    model.eval()

    with torch.no_grad():
        losses, output = model(data, target)

    losses = [torch.mean(loss_value) for loss_value in losses]

    return losses, output


def parse_args():
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
    parser.add_argument('--weights', '-wt', type=str, help='path to latest checkpoint (default: none)', required=True)
    parser.add_argument('--images', '-im', nargs=2, type=str, required=True)
    
    args = parser.parse_args()

    args.model_class = tools.module_to_dict(models)[args.model]
    args.loss_class = tools.module_to_dict(losses)[args.loss]
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.number_gpus < 0: 
        args.number_gpus = torch.cuda.device_count()

    return args


def main():
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

        block.log("Inference")
        losses, output = inference(data=images, target=target, model=model_and_loss)

        block.log(losses)
        cv2.imshow("Flow", output)
        cv2.waitKey()


if __name__ == '__main__':
    main()

