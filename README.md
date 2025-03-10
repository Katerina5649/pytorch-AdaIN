![Model architecture](model.jpg)

This repository was used to display experiments for EPFL CS-503 project in attempt to obtain a realistic style transfer model.

Before running the code it is necessary to download initial model weights from [decoder.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing)/[vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`.

The data created for this project can be recreated using the commands below or can be found in google drive [google drive]
(https://drive.google.com/drive/folders/1SLUqLdlu__opZD7NwYJXnkOHYAAgqYVR?usp=sharing). Please use the following folders with the same names:
- ```input/EPFL_stylized```
- ```input/EPFL_landmark```
- ```input/EPFL_styles```

In the first step of our project we had to create our stylized dataset by applying styles to different landmark images.

To download landmark images from Google Landmarks Dataset you can use the following code in data_extraction/get_landmark_imgs:

```
CUDA_VISIBLE_DEVICES=<gpu_id> python get_landmarks.py
```
This code will download the landmark images for train, validation and test sets and save into ```/input/EPFL_landmark/train/```, ```/input/EPFL_landmark/val/``` and ```/input/EPFL_landmark/test/```. The images can also be downloaded from our drive under ```/input/EPFL_landmark```.

!!!!!! add landmark-style pairing!!!!!

For creating stylized landmark images for test, train and validation sets we used the command below with different keys ```train, val, test ```
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py--key train
```
This code will create stylized landmark images and save into ```input/EPFL_stylized``` folder. We also provide the results of this command in ```input/EPFL_stylized``` on the Google drive.

In the second part of our project we used the stylized images and real landmark images to retrain the model. Pairing of stylized images with real landmark images was done using the following command in ```input/```
```
CUDA_VISIBLE_DEVICES=<gpu_id> python stylized&real_pairing.py
```
This code will pair stylized images with real landmark images to use in training and create the necessary csv files, which are already in ```input/```.

The code below will gather all pairs of stylized images and landmark images and retrain the model. Weights of the retrained model can be found in ```experiments/decoder_iter_81000.pth.tar```
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py 
```

To obtain baseline model performance and retrained model performance on the test set we use the following command:
```
python test_baseline.py --key=test --decoder=<decoder_path>
```
This code will calculate content loss and style loss on test dataset and save in the output folder. To compute the baseline model losses use the option ```--decoder='models/decoder.pth'```. To compute the losses for our retrained model use the option ```--decoder='experiments/decoder_iter_81000.pth.tar'```

To obtain the results used in our user survey we run the following command:
```
python get_survey_imgs.py ```--key='baseline'```
``` 
Providing option ```--key='baseline'``` will create images using the baseline model weights, ```--key='trained'``` will create images using the retrained model weights.

## Downloading style images and generating landmark-style pairs

To create the landmark-style pairs train, test and split datasets:
* Download `train.zip`, `test.zip` and `replacement_for_corrupted_files.zip` from [Kaggle Painter by Numbers](https://www.kaggle.com/competitions/painter-by-numbers/overview)
* Unzip these folders and put all images in the `imgs` folder
* Run the `data_extraction/get_landmark_style_pairs/get_landmark_style_pairs.py` script to generate the pairs, export them as csv files for each split and move the images into the appropriate folders (`train`, `val` or `test`).

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
- [3] T. Weyand, A. Araujo, B. Cao, and J. Sim, “Google landmarks dataset v2 - A large-scale benchmark for instance-level recognition and retrieval,” CoRR, vol. abs/2004.01804, 2020. [Online]. Available: https://github.com/cvdfoundation/google-landmark
