import openpifpaf
import torch
import argparse
import copy
import logging
import torch.multiprocessing as mp
import csv
from default_params import *
from algorithms import *
from helpers import last_ip
import os
import matplotlib.pyplot as plt

try:
    mp.set_start_method('spawn')

except RuntimeError:
    pass


class FallDetector:
    def __init__(self, t=DEFAULT_CONSEC_FRAMES):
        self.consecutive_frames = t
        self.args = self.cli()

    def cli(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        # TODO: Verify the args since they were changed in v0.10.0
        openpifpaf.decoder.cli(parser, force_complete_pose=True,
                               instance_threshold=0.2, seed_threshold=0.5)
        openpifpaf.network.cli(parser)
        parser.add_argument('--resolution', default=0.4, type=float,
                            help=('Resolution prescale factor from 640x480. '
                                  'Will be rounded to multiples of 16.'))
        parser.add_argument('--resize', default=None, type=str,
                            help=('Force input image resize. '
                                  'Example WIDTHxHEIGHT.'))
        parser.add_argument('--num_cams', default=1, type=int,
                            help='Number of Cameras.')
        parser.add_argument('--video', default=None, type=str,
                            help='Path to the video file.\nFor single video fall detection(--num_cams=1), save your videos as abc.xyz and set --video=abc.xyz\nFor 2 video fall detection(--num_cams=2), save your videos as abc1.xyz & abc2.xyz and set --video=abc.xyz')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='debug messages and autoreload')
        parser.add_argument('--disable_cuda', default=False, action='store_true',
                            help='disables cuda support and runs from gpu')

        vis_args = parser.add_argument_group('Visualisation')
        vis_args.add_argument('--plot_graph', default=False, action='store_true',
                              help='Plot the graph of features extracted from keypoints of pose.')
        vis_args.add_argument('--joints', default=True, action='store_true',
                              help='Draw joint\'s keypoints on the output video.')
        vis_args.add_argument('--skeleton', default=True, action='store_true',
                              help='Draw skeleton on the output video.')
        vis_args.add_argument('--coco_points', default=False, action='store_true',
                              help='Visualises the COCO points of the human pose.')
        vis_args.add_argument('--save_output', default=False, action='store_true',
                              help='Save the result in a video file. Output videos are saved in the same directory as input videos with "out" appended at the start of the title')
        vis_args.add_argument('--fps', default=18, type=int,
                              help='FPS for the output video.')
        vis_args.add_argument('--out-path', default='result.avi', type=str,
                              help='Save the output video at the path specified. .avi file format.')

#         vis_args.add_argument('--input_frame', default=False, action='store_true',
#                               help='Save the result in a video file. Output videos are saved in the same directory as input videos with "out" appended at the start of the title')

        vis_args.add_argument('--input_direct', default=None, type=str,
                              help='Save the input link to images directory.')


        args = parser.parse_args()

        # Log
        logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

        # Add args.device
        args.device = torch.device('cpu')
        args.pin_memory = False
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
            args.pin_memory = True

        if args.checkpoint is None:
            args.checkpoint = 'shufflenetv2k16w'

        return args

    def begin(self):
        print('Begin...')
        e = mp.Event()
        queues = [mp.Queue() for _ in range(self.args.num_cams)]
        counter1 = mp.Value('i', 0)
        counter2 = mp.Value('i', 0)
        argss = [copy.deepcopy(self.args) for _ in range(self.args.num_cams)]
        if self.args.num_cams == 1:
            if self.args.video is None:
                argss[0].video = 0
            process1 = mp.Process(target=extract_keypoints_parallel,
                                  args=(queues[0], argss[0], counter1, counter2, self.consecutive_frames, e))
            
            print("Start process1")

            # ARGS:  Namespace(basenet=None, caf_seeds=False, caf_th=0.1, 
            # checkpoint='shufflenetv2k16w', cif_th=0.1, coco_points=False, 
            # connection_method='blend', cross_talk=0.0, debug=False, decoder_workers=None, 
            # dense_connections=False, dense_coupling=0.01, device=device(type='cuda'), 
            # disable_cuda=False, download_progress=True, force_complete_pose=True, 
            # fps=18, greedy=False, head_dropout=0.0, head_quad=1, headnets=None, 
            # instance_threshold=0.2, joints=True, keypoint_threshold=None, 
            # multi_scale=False, multi_scale_hflip=True, num_cams=1, 
            # out_path='result.avi', pin_memory=True, plot_graph=False, pretrained=True, 
            # profile_decoder=None, resize=None, resolution=0.4, save_output=True, 
            # seed_threshold=0.5, skeleton=True, two_scale=False, 
            # video='/content/HumanFallDetection/vids/input/fallingdown.mp4')

            process1.start()
            if self.args.coco_points:
                print("\n\nFALL_DETECTOR.PY, COCO_points - line 120")
                process1.join()
            else:
                print("\n\nFALL_DETECTOR.PY, call mp.Process() class - line 123")
                process2 = mp.Process(target=alg2_sequential, args=(queues, argss,
                                                                    self.consecutive_frames, e))
                process2.start()

            
            print("\n\nFALL_DETECTOR.PY, call process1.join() - line 129")
            process1.join()

        elif self.args.num_cams == 2:
            if self.args.video is None:
                argss[0].video = 0
                argss[1].video = 1
            else:
                try:
                    vid_name = self.args.video.split('.')
                    argss[0].video = ''.join(vid_name[:-1])+'1.'+vid_name[-1]
                    argss[1].video = ''.join(vid_name[:-1])+'2.'+vid_name[-1]
                    print('Video 1:', argss[0].video)
                    print('Video 2:', argss[1].video)
                except Exception as exep:
                    print('Error: argument --video not properly set')
                    print('For 2 video fall detection(--num_cams=2), save your videos as abc1.xyz & abc2.xyz and set --video=abc.xyz')
                    return
            process1_1 = mp.Process(target=extract_keypoints_parallel,
                                    args=(queues[0], argss[0], counter1, counter2, self.consecutive_frames, e))
            process1_2 = mp.Process(target=extract_keypoints_parallel,
                                    args=(queues[1], argss[1], counter2, counter1, self.consecutive_frames, e))
            process1_1.start()
            process1_2.start()
            if self.args.coco_points:
                process1_1.join()
                process1_2.join()
            else:
                process2 = mp.Process(target=alg2_sequential, args=(queues, argss,
                                                                    self.consecutive_frames, e))
                process2.start()
            process1_1.join()
            process1_2.join()
        else:
            print('More than 2 cameras are currently not supported')
            return
        print("\n\nAFTER PROCESSING")

        if not self.args.coco_points:
            process2.join()
        print('Ending...')
        return


if __name__ == "__main__":
    f = FallDetector()
    f.begin()
