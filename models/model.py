import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.backbone import resnet18 as resnet18
from models.backbone import resnet34 as resnet34
from models.abri_backbone import resnet18 as abri_resnet18
from models.abri_backbone import resnet34 as abri_resnet34

########### Change the names of the pre-trained uni-modal encoders here ##############
model_path = {
    'vggsound': 'pretrained_encoder/resnet18_VGGSound.pth',
    'audioset': 'pretrained_encoder/resnet18_AudioSet.pth'
}

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output

class AudioClassifier(nn.Module):
    def __init__(self, args, imagenet_pretrain = False):
        super(AudioClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'ESC50':
            n_classes = 50
        elif args.dataset == 'DCASE':
            n_classes = 15
        elif args.dataset == 'caltech':
            n_classes = 257
        elif args.dataset == 'Audioset':
            n_classes = 527
        elif args.dataset == 'VGGSound_Sub':
            n_classes = 309
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if args.model == 'resnet18':
            self.audio_net = resnet18(imagenet_pretrained = imagenet_pretrain)
            self.fc = nn.Linear(512, n_classes)

        elif args.model == 'resnet34':
            self.audio_net = resnet34(imagenet_pretrained = imagenet_pretrain)
            self.fc = nn.Linear(512, n_classes)

        elif args.model == 'resnet18_abri':
            self.audio_net = abri_resnet18(imagenet_pretrained= imagenet_pretrain, fusion=args.fusion, fus_weight= args.alpha)
            self.fc = nn.Linear(512, n_classes)

        elif args.model == 'resnet34_abri':
            self.audio_net = abri_resnet34(imagenet_pretrained= imagenet_pretrain, fusion=args.fusion, fus_weight= args.alpha)
            self.fc = nn.Linear(512, n_classes)

    def forward(self, audio):

        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        out = self.fc(a)

        return out

    def get_representation(self, audio):

        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        return a

class VisualClassifier(nn.Module):
    def __init__(self, args, imagenet_pretrain = False):
        super(VisualClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'UCF101':
            n_classes = 101
        elif args.dataset == 'ESC50':
            n_classes = 50
        elif args.dataset == 'DCASE':
            n_classes = 15
        elif args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'caltech':
            n_classes = 257
        elif args.dataset == 'VGGSound_Sub':
            n_classes = 309
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.dataset = args.dataset

        if args.model == 'resnet18':
            self.visual_net = resnet18(imagenet_pretrained= imagenet_pretrain)
            self.fc = nn.Linear(512, n_classes)
        elif args.model == 'resnet34':
            self.visual_net = resnet34(imagenet_pretrained= imagenet_pretrain)
            self.fc = nn.Linear(512, n_classes)
        elif args.model == 'resnet18':
            self.visual_net = abri_resnet18(imagenet_pretrained= imagenet_pretrain, fusion=args.fusion, fus_weight= args.alpha)
            self.fc = nn.Linear(512, n_classes)
        elif args.model == 'resnet34':
            self.visual_net = abri_resnet34(imagenet_pretrained= imagenet_pretrain, fusion=args.fusion, fus_weight= args.alpha)
            self.fc = nn.Linear(512, n_classes)

    def forward(self, visual):

        if self.dataset == 'KineticSound' or self.dataset == 'VGGSound_Sub':
            (B, C, T, H, W) = visual.size()
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            visual = visual.view(B * T, C, H, W)
        
        v = self.visual_net(visual)

        if self.dataset == 'KineticSound' or self.dataset == 'VGGSound_Sub':
            (_, C, H, W) = v.size()
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

        else:
            v = F.adaptive_avg_pool2d(v, 1)
            v = torch.flatten(v, 1)

        out = self.fc(v)

        return out


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'VGGSound-Sub':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'UCF101':
            n_classes = 101
        elif args.dataset == 'VGGSound_Sub':
            n_classes = 309
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.fusion_module = ConcatFusion(output_dim=n_classes)

        if args.audio_pretrain == 'imagenet':
            self.audio_net = resnet18(imagenet_pretrained = True)
            print('audio imagenet pretrained')
        elif args.audio_pretrain != 'random':
            self.audio_net = resnet18(imagenet_pretrained = False)
            loaded_dict = torch.load(model_path[args.audio_pretrain])
            state_dict = loaded_dict['model']
            self.audio_net.load_state_dict(state_dict,strict=True)
            print('audio load', model_path[args.audio_pretrain])
        else:
            self.audio_net = resnet18(imagenet_pretrained = False)
            print('audio random')

        if args.visual_pretrain == 'imagenet':
            self.visual_net = resnet18(imagenet_pretrained = True)
            print('visual imagenet pretrained')
        elif args.visual_pretrain != 'random':
            self.visual_net = resnet18(imagenet_pretrained = False)
            loaded_dict = torch.load(model_path[args.visual_pretrain])
            state_dict = loaded_dict['model']
            self.visual_net.load_state_dict(state_dict,strict=True)
            print('visual load', model_path[args.visual_pretrain])
        else:
            self.visual_net = resnet18(imagenet_pretrained = False)
            print('visual random')

        

    def forward(self, audio, visual):
        (B, C, T, H, W) = visual.size()
        visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        visual = visual.view(B * T, C, H, W)


        a = self.audio_net(audio)
        v = self.visual_net(visual)
        
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)
        a = F.adaptive_avg_pool2d(a, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out

class AVClassifier_ft(nn.Module):
    def __init__(self, args):
        super(AVClassifier_ft, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'UCF101':
            n_classes = 101
        elif args.dataset == 'vox1':
            n_classes = 446
        elif args.dataset == 'ssw':
            n_classes = 60
        elif args.dataset == 'VGGSound_Sub':
            n_classes = 309
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.fusion_module = ConcatFusion(output_dim=n_classes)

        if args.dataset == 'KineticSound':
            if args.audio_pretrain == 'audioset':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/audio_audioset.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            elif args.audio_pretrain == 'imagenet':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/audio_imagenet.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            elif args.audio_pretrain == 'vggsound':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/audio_vggsound.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            else:
                self.audio_net = resnet18(imagenet_pretrained = False)
                print('audio random')

            if args.visual_pretrain == 'imagenet':
                self.visual_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/visual_imagenet.pth')
                state_dict = loaded_dict['model']
                self.visual_net.load_state_dict(state_dict,strict=True)

            else:
                self.visual_net = resnet18(imagenet_pretrained = False)
                print('visual imagenet')

        elif args.dataset == 'VGGSound_Sub':
            if args.audio_pretrain == 'audioset':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('VGGSound_Sub/audio_audioset.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            elif args.audio_pretrain == 'imagenet':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('VGGSound_Sub/audio_imagenet.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            else:
                self.audio_net = resnet18(imagenet_pretrained = False)
                print('audio random')

            if args.visual_pretrain == 'imagenet':
                self.visual_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('VGGSound_Sub/visual_imagenet.pth')
                state_dict = loaded_dict['model']
                self.visual_net.load_state_dict(state_dict,strict=True)

            else:
                self.visual_net = resnet18(imagenet_pretrained = True)
                print('visual imagenet')

        

    def forward(self, audio, visual):
        (B, C, T, H, W) = visual.size()
        visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        visual = visual.view(B * T, C, H, W)


        a = self.audio_net(audio)
        v = self.visual_net(visual)
        
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)
        a = F.adaptive_avg_pool2d(a, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class AVClassifier_ft_ABRi(nn.Module):
    def __init__(self, args):
        super(AVClassifier_ft_ABRi, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'UCF101':
            n_classes = 101
        elif args.dataset == 'VGGSound_Sub':
            n_classes = 309
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.fusion_module = ConcatFusion(output_dim=n_classes)

        if args.dataset == 'KineticSound':
            if args.audio_pretrain == 'audioset':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/audio_audioset.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            elif args.audio_pretrain == 'imagenet':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/audio_imagenet.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            elif args.audio_pretrain == 'vggsound':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('KineticSound/audio_vggsound.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            else:
                self.audio_net = resnet18(imagenet_pretrained = False)
                print('audio random')

            if args.visual_pretrain == 'imagenet':
                self.visual_net = abri_resnet18(imagenet_pretrained= False, fusion=args.fusion, fus_weight= args.alpha)
                loaded_dict = torch.load('KineticSound/visual_imagenet-abri.pth')
                state_dict = loaded_dict['model']
                self.visual_net.load_state_dict(state_dict,strict=True)

            else:
                self.visual_net = abri_resnet18(imagenet_pretrained= False, fusion=args.fusion, fus_weight= args.alpha)
                print('visual imagenet')

        

        elif args.dataset == 'VGGSound_Sub':
            if args.audio_pretrain == 'audioset':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('VGGSound_Sub/audio_audioset.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            elif args.audio_pretrain == 'imagenet':
                self.audio_net = resnet18(imagenet_pretrained = False)
                loaded_dict = torch.load('VGGSound_Sub/audio_imagenet.pth')
                state_dict = loaded_dict['model']
                self.audio_net.load_state_dict(state_dict,strict=True)

            else:
                self.audio_net = resnet18(imagenet_pretrained = False)
                print('audio random')

            if args.visual_pretrain == 'imagenet':
                self.visual_net = abri_resnet18(imagenet_pretrained= False, fusion=args.fusion, fus_weight= args.alpha)
                loaded_dict = torch.load('VGGSound_Sub/visual_imagenet-abri.pth')
                state_dict = loaded_dict['model']
                self.visual_net.load_state_dict(state_dict,strict=True)

            else:
                self.visual_net = abri_resnet18(imagenet_pretrained= False, fusion=args.fusion, fus_weight= args.alpha)
                print('visual random')

        

    def forward(self, audio, visual):
        (B, C, T, H, W) = visual.size()
        visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        visual = visual.view(B * T, C, H, W)


        a = self.audio_net(audio)
        v = self.visual_net(visual)
        
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)
        a = F.adaptive_avg_pool2d(a, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out