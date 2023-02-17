import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AVDataset_KS_Mask(Dataset):
  def __init__(self, mode='train', args = None):
    self.data = []
    self.label = []

    self.mode=mode
    
    self.train_txt = '/data/ks/train_1fps_path.txt'
    self.val_txt = '/data/ks/val_1fps_path.txt'
    self.test_txt = '/data/ks/test_1fps_path.txt'
    if mode == 'train':
      csv_file = self.train_txt
      self.audio_path = '/data/ks/ks_audio/train/'
      self.visual_path = '/data/ks/ks_visual/train/'
      if args.mask != 'none':
        if args.model == 'resnet18_abri' or args.model == 'resnet34_abri':
            if args.audio_pretrain == 'audioset':
                self.a_confidence = np.load('/log/confidence/audio/audioset-KineticSound-abri/mean.npy', allow_pickle=True).item()
            elif args.audio_pretrain == 'imagenet':
                self.a_confidence = np.load('/log/confidence/audio/imagenet-KineticSound-abri/mean.npy', allow_pickle=True).item()
            else:
                self.a_confidence = np.load('/log/confidence/audio/vggsound-KineticSound-abri/mean.npy', allow_pickle=True).item()
            self.v_confidence = np.load('/log/confidence/visual/imagenet-KineticSound-abri/mean.npy', allow_pickle=True).item()
        else:
            if args.audio_pretrain == 'audioset':
                self.a_confidence = np.load('/log/confidence/audio/audioset-KineticSound/mean.npy', allow_pickle=True).item()
            elif args.audio_pretrain == 'imagenet':
                self.a_confidence = np.load('/log/confidence/audio/imagenet-KineticSound/mean.npy', allow_pickle=True).item()
            else:
                self.a_confidence = np.load('/log/confidence/audio/vggsound-KineticSound/mean.npy', allow_pickle=True).item()
            self.v_confidence = np.load('/log/confidence/visual/imagenet-KineticSound/mean.npy', allow_pickle=True).item()

    elif mode == 'val':
      csv_file = self.val_txt
      self.audio_path = '/data/ks/ks_audio/val/'
      self.visual_path = '/data/ks/ks_visual/val/'
    else:
      csv_file = self.test_txt
      self.audio_path = '/data/ks/ks_audio/test/'
      self.visual_path = '/data/ks/ks_visual/test/'


    with open(csv_file) as f:
      for line in f:
        item = line.split("\n")[0].split(" ")
        name = item[0].split("/")[-1]
        self.data.append(name)
        self.label.append(int(item[-1]))

    # compute mask ratio
    if mode == 'train':
        self.a_mask_ratio = []
        self.v_mask_ratio = []
        for item in self.data:
            if args.mask == 'conf' or args.mask == 'mean':
                a_conf = self.a_confidence[item]
                v_conf = self.v_confidence[item]

                if v_conf >= args.threshold:
                    a_mask_sample = np.tanh(args.eta_audio * (a_conf - v_conf)) * args.rho_audio
                    v_mask_sample = 0.0

                elif a_conf >= args.threshold:
                    a_mask_sample = 0.0
                    v_mask_sample = np.tanh(args.eta_visual * (v_conf - a_conf)) * args.rho_visual

                else:
                    a_mask_sample = 0.0
                    v_mask_sample = 0.0

            else:
                a_mask_sample = 0.0
                v_mask_sample = 0.0

            self.a_mask_ratio.append(a_mask_sample)
            self.v_mask_ratio.append(v_mask_sample)
        print('mean mask_a:',np.mean(self.a_mask_ratio),'mean mask_v:',np.mean(self.v_mask_ratio))

        if args.mask == 'mean':
            print('now use mean mask')
            mean_a = np.mean(self.a_mask_ratio)
            mean_v = np.mean(self.v_mask_ratio)
            for i in range(len(self.a_mask_ratio)):
                self.a_mask_ratio [i] = mean_a
                self.v_mask_ratio [i] = mean_v


    print('data load over')
    
    self.class_num = 31

    print('# of files = %d ' % len(self.data))

    #Audio
        # self.audio_conf = audio_conf
    self.melbins = 128

    self.norm_mean = -4.503877
    self.norm_std = 5.141276
    # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
    # set it as True ONLY when you are getting the normalization stats.
    self.skip_norm = False
    if self.skip_norm:
        print('now skip normalization (use it ONLY when you are computing the normalization stats).')
    else:
        print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))


  def __len__(self):
    return len(self.data)

  def add_mask_audio(self, spec, ratio):
    # patch size
    patch_w = 32
    patch_l = 8

    w_num = int(1024 / patch_w)
    l_num = int(128 / patch_l)
    total_num = w_num * l_num
    patch_num = int(total_num * ratio)
    patch_list = np.random.choice(total_num, patch_num, replace=False)

    for index in patch_list:
        patch_x = index % w_num * patch_w
        patch_y = int(index / w_num) * patch_l
        spec[patch_x:patch_x+patch_w, patch_y:patch_y+patch_l] = 0.0

    return spec

  def add_mask_visual(self, image, ratio):
    # patch size
    patch_w = 10
    patch_l = 10

    w_num = int(224 / patch_w)
    l_num = int(224 / patch_l)
    total_num = w_num * l_num
    patch_num = int(total_num * ratio)
    patch_list = np.random.choice(total_num, patch_num, replace=False)

    for index in patch_list:
        patch_x = index % w_num * patch_w
        patch_y = int(index / w_num) * patch_l   
        image[:, patch_x:patch_x+patch_w, patch_y:patch_y+patch_l] = 0.0

    return image

  
  def __getitem__(self, index):
    """
        returns: name, image, fbank, label
        where image is a FloatTensor of size (C, T, H, W)
        fbank is a FloatTensor of size (Time_bin, Freq_bin) for spectrogram
        name is a string
        """
    datum = self.data[index]

    if self.mode == 'train':
        a_m_ratio = self.a_mask_ratio[index]
        v_m_ratio = self.v_mask_ratio[index]

    # Audio
    # Borrow from "AST: Audio Spectrogram Transformer"
    # https://github.com/YuanGongND/ast
    waveform, sr = torchaudio.load(self.audio_path + datum + '.wav')
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

    target_length = 1024
    n_frames = fbank.shape[0]
    # print(n_frames)
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

    if self.mode == 'train' and not self.skip_norm:
        if a_m_ratio > 0:
            fbank = self.add_mask_audio(fbank, a_m_ratio)

    # Visual
    if self.mode == 'train':
        transf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    folder_path = self.visual_path + datum
    file_num = len(os.listdir(folder_path))
    pick_num = 3
    seg = int(file_num/pick_num)
    image_arr = []

    for i in range(pick_num):
      if self.mode == 'train':
        chosen_index = random.randint(i*seg + 1, i*seg + seg)
      else:
        chosen_index = i*seg + int(seg/2)
      path = folder_path + '/frame_0000' + str(chosen_index) + '.jpg'
      tranf_image = transf(Image.open(path).convert('RGB'))
      if self.mode == 'train' and v_m_ratio > 0:
        tranf_image = self.add_mask_visual(tranf_image, v_m_ratio)
      image_arr.append(tranf_image.unsqueeze(0))

    images = torch.cat(image_arr)

    return self.data[index], fbank, images, self.label[index]