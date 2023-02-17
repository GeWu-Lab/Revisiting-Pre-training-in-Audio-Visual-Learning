import csv
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import os


class Audio_KS(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.audio_path = '/home/ruoxuan_feng/ks_audio/train/'
            self.datapath = '/home/ruoxuan_feng/ks_audio/train_1fps_path.txt'
        elif mode == 'val':
            self.audio_path = '/home/ruoxuan_feng/ks_audio/val/'
            self.datapath = '/home/ruoxuan_feng/ks_audio/val_1fps_path.txt'
        else:
            self.audio_path = '/home/ruoxuan_feng/ks_audio/test/'
            self.datapath = '/home/ruoxuan_feng/ks_audio/test_1fps_path.txt'
        self.data = []
        self.label = []
        self.mode = mode
        with open(self.datapath) as f:
          for line in f:
            item = line.split("\n")[0].split(" ")
            name = item[0].split("/")[-1]
            self.data.append(name)
            self.label.append(int(item[-1]))

        self.class_num = 31
        self.melbins = 128

        # Audio
        # Borrow from "AST: Audio Spectrogram Transformer"
        # https://github.com/YuanGongND/ast
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = -4.503877
        self.norm_std = 5.141276
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))


        # self.index_dict = make_index_dict(label_csv)
        # self.label_num = len(self.index_dict)
        # print('number of classes is {:d}'.format(self.label_num))

    def __getitem__(self, index):
        """
        returns: name, fbank, label
        where fbank is a FloatTensor of size (Time_bin, Freq_bin) for spectrogram
        name is a string
        """
        datum = self.data[index]

        waveform, sr = torchaudio.load(self.audio_path + datum + '.wav')
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

            # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        # the output fbank shape is [Time_bin, Freq_bin], e.g., [1024, 128]
        return self.data[index], fbank, self.label[index]

    def __len__(self):
        return len(self.data)

class Audio_ESC50(Dataset):
    def __init__(self, mode='train', eval_fold = '5'):
        self.audio_path = '/home/ruoxuan_feng/ESC-50/ESC-50-master/audio/'
        self.datapath = '/home/ruoxuan_feng/ESC-50/ESC-50-master/meta/esc50.csv'
        self.data = []
        self.label = []
        self.mode = mode
        with open(self.datapath) as f:
          csv_reader = csv.reader(f)
          next(csv_reader)
          for item in csv_reader:
            if item[1] != eval_fold and mode == 'train':
              self.data.append(item[0])
              self.label.append(int(item[2]))
            elif item[1] == eval_fold and mode == 'test':
              self.data.append(item[0])
              self.label.append(int(item[2]))

        print('data num: ', len(self.data))

        self.class_num = 50
        self.melbins = 128
        if self.mode == 'train':
            self.freqm = 15
            self.timem = 64
        else:
            self.freqm = 0
            self.timem = 0
        print('now using following mask: {:d} freq, {:d} time'.format(self.freqm, self.timem))

        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = -5.902779
        self.norm_std = 4.507141
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))


    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(self.audio_path + filename)
        waveform = torch.tile(waveform, (1, 2))
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank


    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)

        datum = self.data[index]
        label_indices = np.zeros(self.class_num)
        fbank = self._wav2fbank(datum)
        label_indices[self.label[index]] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        if self.mode == 'train' and not self.skip_norm:
          freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
          timem = torchaudio.transforms.TimeMasking(self.timem)
          fbank = torch.transpose(fbank, 0, 1)
          fbank = fbank.unsqueeze(0)
          if self.freqm != 0:
              fbank = freqm(fbank)
          if self.timem != 0:
              fbank = timem(fbank)
          fbank = fbank.squeeze(0)
          fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        else:
            pass

        return fbank, label_indices

    def __len__(self):
        return len(self.data)