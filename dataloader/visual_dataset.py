import os
import torch
import csv
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2

class Visual_KS(Dataset):
    def __init__(self, mode):
      self.data = []
      self.label = []

      self.mode = mode
      
      self.train_txt = '/data/ks/train_1fps_path.txt'
      self.val_txt = '/data/ks/val_1fps_path.txt'
      self.test_txt = '/data/ks/test_1fps_path.txt'

      if mode == 'train':
          csv_file = self.train_txt
          self.visual_path = '/data/ks/ks_visual/train/'
      elif mode == 'val':
          csv_file = self.val_txt
          self.visual_path = '/data/ks/ks_visual/val/'
      else:
          csv_file = self.test_txt
          self.visual_path = '/data/ks/ks_visual/test/'

      with open(csv_file) as f:
        for line in f:
          item = line.split("\n")[0].split(" ")
          name = item[0].split("/")[-1]
          self.data.append(name)
          self.label.append(int(item[-1]))

      self.class_num = 31

      print('# of files = %d ' % len(self.data))

      print('# of classes = %d' % self.class_num)


    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      datum = self.data[idx]

      #Visual
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
          image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))

      images = torch.cat(image_arr)

      return self.data[idx], images, self.label[idx]

class Visual_Caltech(Dataset):
    def __init__(self, mode):
        self.data = []
        self.labels = []
        self.mode = mode
        
        self.train_txt = '/data/caltech256/caltech256_train.csv'
        self.test_txt = '/data/caltech256/caltech256_test.csv'
        self.data_path = '/data/caltech256/256_ObjectCategories/'

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                name = item[0]
                label = item[1]
                self.data.append(name)
                self.labels.append(int(label))

        print('data load over')
        print("now in", self.mode)
        self._init_atransform()
        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data[idx]

        path = self.data_path + img_file
        
        if self.mode == 'train':
            img = Image.open(path).convert('RGB')
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale= (1,1), ratio=(1,1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            t_img = transf(img)
            
        else:
            img = cv2.imread(path)
            h,w,_ = img.shape
            scale = 256/min(h,w)
            img = cv2.resize(img,(int(w*scale),int(h*scale)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            t_img = transf(img)

        return t_img, self.labels[idx]