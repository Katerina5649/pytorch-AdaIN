import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper
import pandas as pd
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, key = 'train', transform = None):
        super(FlatFolderDataset, self).__init__()
        self.key = key
        self.transform = transform
        if key == 'train':   
            df = pd.read_csv('input/train_pairs.csv', sep = ',') 
        else:
            df = pd.read_csv('input/valid_pairs.csv', sep = ',') 
        self.landmarks = df['stylized_name']
        self.styles = df['paired_img']


    def __getitem__(self, index):
        
        landmark = Image.open(str(f"input/EPFL_stylized/{self.key}/{self.landmarks[index]}")).convert('RGB')
        landmark = self.transform(landmark)

        style = f"input/EPFL_landmark/{self.key}/{self.styles[index]}"
        style = Image.open(style).convert('RGB')
        style = self.transform(style)
        return landmark, style

    def __len__(self):
        return len(self.landmarks)

    def name(self):
        return f'FlatFolderDataset_{key}'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
#parser.add_argument('--content_dir', type=str, required=True,
#                    help='Directory path to a batch of content images')
#parser.add_argument('--style_dir', type=str, required=True,
#                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=1000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
decoder.load_state_dict(torch.load(args.decoder))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

def get_batch(loader):
  is_err = True
  while is_err:
    try:
      test_content_images, test_style_images = next(loader)
      is_err = False
    except Exception as e:
      is_err = True
  return test_content_images, test_style_images

train_dataset =  FlatFolderDataset(key = "train", transform = train_transform())
val_dataset = FlatFolderDataset(key = "val", transform = train_transform())

train_content_iter = iter(data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(train_dataset),
    num_workers=args.n_threads))

val_content_iter = iter(data.DataLoader(
    train_dataset, batch_size=args.batch_size * 8,
    sampler=InfiniteSamplerWrapper(train_dataset),
    num_workers=args.n_threads))

test_content_images, test_style_images = get_batch(val_content_iter)

test_content_images, test_style_images = test_content_images.to(device), test_style_images.to(device)
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    
    content_images, style_images = get_batch(train_content_iter)

    content_images, style_images = content_images.to(device), style_images.to(device)
    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    train_loss = loss_c + loss_s

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    writer.add_scalar('train_loss_content', loss_c.item(), i + 1)
    writer.add_scalar('train_loss_style', loss_s.item(), i + 1)

    
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
      with torch.no_grad():
        
        content_images, style_images = get_batch(val_content_iter)

        content_images, style_images = content_images.to(device), style_images.to(device)
        loss_c, loss_s = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        val_loss = loss_c + loss_s
        print(f"train loss {train_loss.item()} val loss {val_loss}")

        writer.add_scalar('val_loss_content', loss_c.item(), i + 1)
        writer.add_scalar('val_loss_style', loss_s.item(), i + 1)
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))
writer.close()
