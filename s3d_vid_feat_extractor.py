"""
https://github.com/antoine77340/video_feature_extractor/blob/master/extract.py
https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/youcook_loader.py

larger batch size (>1) needs rewrite collate_fn
"""

import torch as th
import math
import numpy as np
from torch.utils.data import DataLoader
import argparse
from video_feature_extractor.random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F

from S3D_HowTo100M.s3dg import S3D

from torch.utils.data import Dataset
import pandas as pd
import os
import ffmpeg
import subprocess


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            csv,
            framerate=16,
            size=224,
            num_frames=16,
            center_crop=True,
            crop_only=False
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.size = size
        self.framerate = framerate
        self.num_frames = num_frames
        self.num_sec = self.num_frames / float(self.framerate)
        self.center_crop = center_crop
        self.crop_only = crop_only

    def __len__(self):
        return len(self.csv)

    def _get_duration(self, video_path):
        """
        https://stackoverflow.com/questions/31024968/using-ffmpeg-to-obtain-video-durations-in-python
        Get the duration of a video using ffprobe.
        """
        cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(video_path)
        output = subprocess.check_output(
            cmd,
            shell=True, # Let this run in the shell
            stderr=subprocess.STDOUT
        )
        # return round(float(output))  # ugly, but rounds your seconds up or down
        return float(output)

    def _get_video_dim(self, video_path):
        """ fail on wbem format, it works for height & width """
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        duration = float(video_stream['duration'])
        return duration

    def _get_video_start(self, video_path, start):
        """
        """
        start_seek = start
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec)
            .filter('fps', fps=self.framerate)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]


    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]

        if not(os.path.isfile(output_file)) and os.path.isfile(video_path):
            print('Decoding video: {}'.format(video_path))
            try:
                duration = self._get_duration(video_path)
            except:
                print('ffprobe failed at: {}'.format(video_path))
                return {'video': th.zeros(1), 'input': video_path,
                        'output': output_file}

            video = th.zeros(int(duration), 3, self.num_frames, self.size, self.size)
            start_ind = np.linspace(0.0, int(duration) - self.num_sec, int(duration))
            for i, s in enumerate(start_ind):
                video[i] = self._get_video_start(video_path, s)
        else:
            video = th.zeros(1)
            
        return {'video': video, 'input': video_path, 'output': output_file}



def parse_args():

    parser = argparse.ArgumentParser(description='Easy video feature extractor')

    parser.add_argument('--csv', type=str, help='input csv with video input path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--half_precision', type=int, default=1, help='output half precision float')
    parser.add_argument('--num_decoding_thread', type=int, default=4, help='Num parallel thread for video decoding')
    parser.add_argument('--l2_normalize', type=int, default=1, help='l2 normalize feature')
    parser.add_argument('--feature_type', type=str, default='mixed_5c', help='mixed_5c | video_embedding')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    feature_type = args.feature_type

    dataset = VideoLoader(
        args.csv,
        framerate=16,
        size=256,
        num_frames=16,
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=1, # args.batch_size
        shuffle=False,
        num_workers=0, # args.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )

    # preprocess = Preprocessing('s3d')

    # Instantiate the model
    model = S3D('S3D_HowTo100M/pretrained_weights/s3d_dict.npy', 512)

    # Load the model weights
    model.load_state_dict(th.load('S3D_HowTo100M/pretrained_weights/s3d_howto100m.pth'))
    model.eval()
    model.cuda()

    with th.no_grad():
        for k, data in enumerate(loader):
            input_file = data['input'][0]
            output_file = data['output'][0]
            if len(data['video'].shape) > 3:
                print('Computing features of video {}/{}: {}'.format(
                    k + 1, n_dataset, input_file))
                video = data['video'].squeeze()
                # if len(video.shape) == 4:
                    # video = preprocess(video)
                video = video / 255.0
                n_chunk = len(video)

                if feature_type == 'mixed_5c':
                    features = th.cuda.FloatTensor(n_chunk, 1024).fill_(0)
                elif feature_type == 'video_embedding':
                    features = th.cuda.FloatTensor(n_chunk, 512).fill_(0)
                else:
                    raise ValueError

                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):                    
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model(video_batch)[feature_type]
                    # if args.l2_normalize:
                    #     batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype('float16')
                np.save(output_file, features)
            else:
                print('Video {} already processed.'.format(input_file))