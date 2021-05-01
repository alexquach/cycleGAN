import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

# A = Photos
# B = Sketches

def perform_inference(dataroot="/content/sketchy/", A2B=True, B2A=False,\
    generator_A2B="output/netG_A2B.pth", generator_B2A="output/netG_B2A.pth",\
    batch_size = 1, input_nc=3, output_nc=3, size=256, cuda=True, n_cpu=8):

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    if A2B:
        netG_A2B = Generator(input_nc, output_nc)
        if cuda:
            netG_A2B.cuda()
        netG_A2B.load_state_dict(torch.load(generator_A2B))
        netG_A2B.eval()
        input_A = Tensor(batch_size, input_nc, size, size)

    if B2A:
        netG_B2A = Generator(output_nc, input_nc)
        if cuda:
            netG_B2A.cuda()
        netG_B2A.load_state_dict(torch.load(generator_B2A))
        netG_B2A.eval()
        input_B = Tensor(batch_size, output_nc, size, size)

    # Dataset loader
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, mode='test'), 
                            batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        if A2B:
            real_A = Variable(input_A.copy_(batch['A']))
            fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
            save_image(fake_B, 'output/B/%04d.png' % (i+1))

        if B2A:
            real_B = Variable(input_B.copy_(batch['B']))
            fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
            save_image(fake_A, 'output/A/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')