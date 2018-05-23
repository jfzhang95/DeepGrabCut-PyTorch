import socket
from datetime import datetime
import os
import glob
from collections import OrderedDict
import scipy.misc as sm
import numpy as np


# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataset import pascalvoc
from mypath import Path
from networks import deeplab_resnet as resnet
from layers.loss import class_balanced_cross_entropy_loss

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
image_width = 450
image_height = 450

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 5  # Training batch size
testBatch = 5  # Testing batch size
useTest = 1  # See evolution of the test set when training?
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 20  # Store a model every snapshot epochs
relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
nInputChannels = 4  # Number of input channels (RGB + Distance Map of bounding box)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-8  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

print(save_dir_root)
print(exp_name)

if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'dextr_pascal'
net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)

if resume_epoch == 0:
    print("Initializing from pretrained Deeplab-v2 model")
else:
    print("Initializing weights from: {}".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU
train_params = [{'params': resnet.get_1x_lr_params(net), 'lr': p['lr']},
                {'params': resnet.get_10x_lr_params(net), 'lr': p['lr'] * 10}]
if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    voc_train = pascalvoc.PascalVocDataset(image_size=(image_width, image_height),
                                             phase='train',
                                             transform=pascalvoc.ToTensor())

    trainloader = DataLoader(dataset=voc_train, batch_size=5)

    # generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)

    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")







