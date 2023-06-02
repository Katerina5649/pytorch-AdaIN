# Downloading style images and generating landmark-style pairs

To create the landmark-style pairs train, test and split datasets:
* Download `train.zip`, `test.zip` and `replacement_for_corrupted_files.zip` from [Kaggle Painter by Numbers](https://www.kaggle.com/competitions/painter-by-numbers/overview)
* Unzip these folders and put all images in the `imgs` folder
* Run the `get_landmark_style_pairs.py` script to generate the pairs, export them as csv files for each split and move the images into the appropriate folders (`train`, `val` or `test`).