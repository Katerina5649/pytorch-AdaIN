![Model architecture](model.jpg)

This repository was used to display experimemts for EPFL CS-503 project in attempt to obtain a realistic style transfer model.

Before running the code it is necessary to download model weights from [decoder.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing)/[vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`.

The data created for this project can be recreated using the commands below or can be found in google drive [google drive]
(https://drive.google.com/drive/folders/1SLUqLdlu__opZD7NwYJXnkOHYAAgqYVR?usp=sharing). Please use the following folders with the same names:
  input/EPFL_stylized
  input/EPFL_landmark
  input/EPFL_styles

In the first step of our project we had to create our stylized dataset by applying styles to different landmark images.

For creating stylized landmark images for test, train and validation sets we used the command below with different keys ```train, val, test ```
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py--key train
```
This code will create stylized landmark images and save into ```input/EPFL_stylized``` folder. We also provide the results of this command in ```input/EPFL_stylized``` on the Google drive.

In the second part of our project we used the stylized images and real landmark images to retrain the model and used the following command:
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py 
```
The code below will gather all pairs of stylized images and  landmark images and retrain the model. Weights of the retrained model can be found in ```experiments/decoder_iter_81000.pth.tar```

To obtain baseline model performance and retrained model performance on the test set we use the following command:

```
python test_baseline.py --key=test --decoder=<decoder_path>
```

This code will calculate content loss and style loss on test dataset and save in the output folder. To compute the baseline model losses use the option ```--decoder='models/decoder.pth'```. To compute the losses for our retrained model use the option ```--decoder='experiments/decoder_iter_81000.pth.tar'```

## Requirements
Please install requirements by `pip install -r requirements.txt`

- Python 3.5+
- PyTorch 0.4+
- TorchVision
- Pillow

(optional, for training)
- tqdm
- TensorboardX





For more details and parameters, please refer to --help option.

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
