![Model architecture](model.jpg)

This repository was used to conduct experimemts for EPFL CS-503 project.

Before running the code it is necessary to download our data from google drive [google drive]
(https://drive.google.com/drive/folders/1SLUqLdlu__opZD7NwYJXnkOHYAAgqYVR?usp=sharing).
Also, you can download model's weight [decoder.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing)/[vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`.

It is important to move all files from ```input``` folder from google drive to local folder.

To apply realitic style transfer firstle we had to create our dataset by applying styles to different landmarks.

For creating test, train and validation datasets we used the command below with different keys ```train, val, test ```
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py--key train
```
This code will create stylized image and save it to the ```input/EPFL_stylized``` folder. We also provide the results of this command in ```input/EPFL_stylized``` on the Google drive.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py 
```
This code will upload all pairs of landmark and stylized images and contuct training.


To test our model and evalute the perfomace we use command 

```
python test_baseline.py --key=test --decoder=<decoder_path>
```

This code will calculate loss for decoder for all test dataset and save style and content loss in the output folder.

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

I share the model trained by this code [here](https://drive.google.com/file/d/1YIBRdgGBoVllLhmz_N7PwfeP5V9Vz2Nr/view?usp=sharing)

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
