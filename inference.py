import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torchvision

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


def perform_single_inference(filepath="/content/sketchy/", output_png_path="/content/sketchy/output/0001.png",\
    A2B=True, B2A=False, A2B2A=True, B2A2B=False,\
    generator_A2B="output/netG_A2B.pth", generator_B2A="output/netG_B2A.pth",\
    batch_size = 1, input_nc=3, output_nc=3, size=256, cuda=True, n_cpu=8):

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    image = torchvision.io.read_image(filepath).float()
    
    transforms_ = torch.nn.Sequential(
        transforms.Resize(256),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    )

    if A2B or B2A2B or A2B2A:
        netG_A2B = Generator(input_nc, output_nc)
        netG_A2B.load_state_dict(torch.load(generator_A2B))
        netG_A2B.eval()
        if cuda:
            netG_A2B.cuda()

    if B2A or B2A2B or A2B2A:
        netG_B2A = Generator(output_nc, input_nc)
        netG_B2A.load_state_dict(torch.load(generator_B2A))
        netG_B2A.eval()
        if cuda:
            netG_B2A.cuda()


    if A2B or A2B2A:
        input_A = transforms_(image).unsqueeze(0)

        if cuda:
            input_A = input_A.cuda()

        real_A = Variable(input_A.detach().clone())
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)

        if A2B2A:
            fake_A = 0.5*(netG_B2A(fake_B).data + 1.0)
            save_image(fake_A, output_png_path)
            return fake_A, output_png_path

        save_image(fake_B, output_png_path)
        return fake_B, output_png_path

    if B2A or B2A2B:
        input_B = transforms_(image).unsqueeze(0)

        if cuda:
            input_B = input_B.cuda()
        
        real_B = Variable(input_B.detach().clone())
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
         
        if B2A2B:
            fake_B = 0.5*(netG_A2B(fake_A).data + 1.0)
            save_image(fake_B, output_png_path)
            return fake_B, output_png_path

        save_image(fake_A, output_png_path)
        return fake_A, output_png_path

    sys.stdout.write('\rGenerated images')
    sys.stdout.write('\n')