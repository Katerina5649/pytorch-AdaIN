import argparse
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import net
from function import adaptive_instance_normalization, coral
from sampler import InfiniteSamplerWrapper


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

parser.add_argument('--key', type=str)

args = parser.parse_args()

import pandas
import os

##path = 'input'
stylized_path = "input/EPFL_stylized/test"  
landmark_path = "input/EPFL_landmark/test"  
if args.key:
  csv_path = "input/test_pairs.csv"
  df = pandas.read_csv(csv_path)
  #csv_path = f"input/EPFL_styles/landmark_style_pairs_{args.key}.csv"
  #df_full = pandas.read_csv(csv_path)
  #df = df_full.iloc[10000:20000]
  #print('10-20 length:', len(df))

#img_path = [ os.path.join(path, img) for img in df['style_img_path'].to_list()]
#landmark_img_path = [ os.path.join(path, img) for img in df['landmark_img_path'].to_list()]


do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#output_dir = Path(args.output)
#output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir or args.key)
if args.content:
    content_paths = [Path(args.content)]
elif args.content_dir:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]
else:
    output_dir = Path("output/baseline/")
    content_paths = [ Path(os.path.join(stylized_path, img)) for img in df['stylized_name'].to_list()]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir or args.key)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
elif args.style_dir:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
else:
    style_paths = [ Path(os.path.join(landmark_path, img)) for img in df['paired_img'].to_list()]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

network = net.Net(vgg, decoder)
network.eval()
network.to(device)

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, True)
style_tf = test_transform(args.style_size, True)


print('All model downloaded')
content_loss=[]
style_loss=[]
for content_path, style_path in tqdm(zip(content_paths, style_paths)):
    #print(content_path, style_path)
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        try:
            #print(os.listdir(output_dir))
            #output_name = output_dir / 'test_baseline_{:s}_with_{:s}{:s}'.format(
             #   content_path.stem, style_path.stem, args.save_ext)
        
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))

            if args.preserve_color:
                style = coral(style, content)
            
            content_iter = iter(data.DataLoader(content.unsqueeze(0), batch_size=1,
                                #sampler=InfiniteSamplerWrapper(content.unsqueeze(0)),
                                num_workers=1))
            style_iter = iter(data.DataLoader(style.unsqueeze(0), batch_size=1,
                                #sampler=InfiniteSamplerWrapper(style.unsqueeze(0)),
                                num_workers=1))
            
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            
            with torch.no_grad():
                  #output = style_transfer(vgg, decoder, content, style, args.alpha)
                  content_images = next(content_iter).to(device)
                  style_images = next(style_iter).to(device)
                  loss_c, loss_s = network(content_images, style_images)
                  content_loss.append(loss_c.detach().cpu().numpy())
                  style_loss.append(loss_s.detach().cpu().numpy())
            #output = output.cpu()

            #print(f'output {output.shape} saved {str(output_name)}')   
            #save_image(output, str(output_name))
            
        except Exception as e: 
            print(f"{e} for {content_path}, {style_path}") 
df = pd.DataFrame({'content_loss': content_loss, 'style_loss': style_loss})
if args.decoder=='models/decoder.pth':
    df.to_csv('./output/test_baseline_loss.csv', index=False)
    print('Test baseline finished')
else:
    df.to_csv('./output/test_trained_loss.csv', index=False)
    print('Test trained finished')
    

