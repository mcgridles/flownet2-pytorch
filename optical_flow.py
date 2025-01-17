import os
import argparse
import numpy as np
import colorama
import cv2
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import models
import losses
from flow_utils import frame_utils, tools


class ModelAndLoss(nn.Module):
    """
    PyTorch model
    """
    
    def __init__(self, args):
        super(ModelAndLoss, self).__init__()
        kwargs = tools.kwargs_from_args(args, 'model')
        self.model = args.model_class(args, **kwargs)

        kwargs = tools.kwargs_from_args(args, 'loss')
        self.loss = args.loss_class(args, **kwargs)

    def forward(self, data):
        output = self.model(data)

        return output


class StaticCenterCrop(object):
    """
    Crops center of image.
    """

    def __init__(self, im_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = im_size

    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2, :]


class OpticalFlow:
    """
    FlowNet2 optical flow inference interface.

    Arguments
        --number_gpus (int) -> Number of GPUs to use, default=-1
        --cuda (bool) -> Attempt to use CUDA if true
        --seed (int) -> RNG seed, default=1
        --rgb_max (float) -> Max RGB value, default=255.0
        --fp16 (bool) -> Run model in pseudo-fp16 mode (fp16 storage fp32 math)
        --fp16_scale (float) -> Loss scaling, positive power of 2 values can improve fp16 convergence, default=1024.0
        --inference_size (int) -> Spatial size divisible by 64, default (-1,-1) - largest possible valid size would be used
        --model (str) -> Model type, default='FlowNet2', options: ChannelNorm, FlowNet2, FlowNet2C, FlowNet2CS,
                         FlowNet2CSS, FlowNet2S, FlowNet2SD, Resample2d, tofp16, tofp32
        --model_div_flow (float) -> Flow division
        --model_batchNorm (bool) -> Perform batch normalization if true
        --loss (str) -> Loss type, default='L1Loss', options: L1, L1Loss, L2, L2Loss, MultiScale
        --optical_weights (str) -> Path to latest weights file
        --images (str) -> Path to two input images

    Example
        >>> args = parser.parse_args()
        >>> of = OpticalFlow(args)
        >>> output = of.run(args.images)
        >>> of.display_flow(output, save_path=os.path.dirname(os.path.abspath(__file__)))
    """

    def __init__(self, args):
        self.args = args
        self.model = None

        self.load()

    def load(self):
        """
        Set up FlowNet2 inference model.
        """
        
        with tools.TimerBlock('Building {} model'.format(self.args.model)) as block:
            model_and_loss = ModelAndLoss(self.args)

            block.log('Number of parameters: {}'.format(
                sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

            # Passing to cuda or wrap with data parallel, model and loss
            if self.args.cuda and (self.args.number_gpus > 0) and self.args.fp16:
                block.log('Parallelizing')
                model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(self.args.number_gpus)))

                block.log('Initializing CUDA')
                model_and_loss = model_and_loss.cuda().half()
                torch.cuda.manual_seed(self.args.seed)
            elif self.args.cuda and self.args.number_gpus > 0:
                block.log('Initializing CUDA')
                model_and_loss = model_and_loss.cuda()

                block.log('Parallelizing')
                model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(self.args.number_gpus)))
                torch.cuda.manual_seed(self.args.seed)
            else:
                block.log('CUDA not being used')
                torch.manual_seed(self.args.seed)

            if os.path.isfile(self.args.optical_weights):
                block.log('Loading weights {}'.format(self.args.optical_weights))
                checkpoint = torch.load(self.args.optical_weights)

                # CUDA weights must be loaded slightly differently
                if next(model_and_loss.parameters()).is_cuda:
                    model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
                else:
                    model_and_loss.model.load_state_dict(checkpoint['state_dict'])

                block.log('Loaded checkpoint {} (at epoch {})'.format(self.args.optical_weights, checkpoint['epoch']))
            else:
                block.log('No checkpoint found at {}'.format(self.args.optical_weights))
                quit()

            self.model = model_and_loss
            self.model.eval()

    def __call__(self, images):
        """
        Run optical flow pipeline on two images.
        """

        images = self.preprocess(images)
        output = self.inference(images)

        output = output.cpu()      # convert to CPU tensor
        output = output.squeeze(0) # remove batch dimension
        output = output.numpy()    # convert to numpy array
        
        return output

    def preprocess(self, images):
        """
        Prepare images for inference and convert to torch tensors.
        """

        frame_size = images[0].shape

        render_size = self.args.inference_size
        if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
            render_size[0] = ((frame_size[0]) // 64) * 64
            render_size[1] = ((frame_size[1]) // 64) * 64

        image_size = frame_size[:2]
        cropper = StaticCenterCrop(image_size, render_size)
        images = list(map(cropper, images))

        # Reshape and convert to tensors
        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))

        return images

    def inference(self, images):
        """
        Perform inference (calculate optical flow) for two images.
        """
    
        images = images.unsqueeze(0)
        if self.args.cuda and self.args.number_gpus > 0:
            images.cuda()
            
        with torch.no_grad():
            output = self.model(images)

        return output

    @staticmethod
    def display_flow(uv, save_path=''):
        """
        Displays optical flow using HSV color space.
        """

        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)

        u = uv[:, :, 0]
        v = uv[:, :, 1]

        hue = (np.arctan2(v, u) + 2 * np.pi) % (2 * np.pi)  # Angles [0, 2*pi]]
        hue = np.interp(hue, [0, 2*np.pi], [0, 179])        # Convert to [0, 179]
        saturation = np.linalg.norm(uv, axis=2) * 255       # Magnitudes [0, 255]
        value = np.ones_like(hue) * 255

        hsv = np.dstack((hue, saturation, value)).astype(np.uint8)

        if save_path:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(save_path, 'flow.jpg'), bgr)
        else:
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            plt.imshow(rgb)
            plt.show()
            
            
def parse_args(model):
    """
    Create argument parser for optical flow class (inference only).

    :return: (argparse.ArgumentParser) -> Parser with arguments for optical flow
    """

    parser = argparse.ArgumentParser(add_help=False)

    # CUDA
    parser.add_argument('--number_gpus', type=int, default=-1, help='Number of GPUs to use')

    # Preprocessing
    parser.add_argument('--seed', type=int, default=1, help='RNG seed')
    parser.add_argument('--rgb_max', type=float, default=255.0, help='Max RGB value')
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.0,
        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
        help='Spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

    # Weights
    parser.add_argument('--optical_weights', type=str, help='Path to FlowNet weights', default='')

    ### Model and loss ###
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default=model)
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    with tools.TimerBlock('Parsing Arguments') as block:
        args, unknown = parser.parse_known_args()
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Print all arguments, color the non-defaults
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.loss_class = tools.module_to_dict(losses)[args.loss]
        args.cuda = torch.cuda.is_available()

    return args
