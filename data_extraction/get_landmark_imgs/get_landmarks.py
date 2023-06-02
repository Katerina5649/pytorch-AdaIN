import numpy as np
import pandas as pd
import os
import urllib
import tarfile
import shutil
import io
import random
from sklearn.model_selection import train_test_split
from urllib.request import Request, urlopen

landmark_pop_filter = 150 #consider landmarks with more than 150 imgs
cat_char_limit = 2 #consider only landmarks with name longer than 2 chars
train_cat_num = 200 # number of landmark categories to include in train set
valid_test_cat_num = 10 # number of landmark categories to only include in validation and test sets
train_per_cat = 25 # number of images per landmark category in train set
valid_test_per_cat = 5 # number of images per landmark category in validation and test sets
tar_file_nums = 500 # number of tar files to download
temp_path = './temp/' # temporary path to extract tar file
train_save_path = '../../input/EPFL_landmark/train/'
valid_save_path = '../../input/EPFL_landmark/valid/'
test_save_path = '../../input/EPFL_landmark/test/'


# Google landmarks v2 dataset (train set)
#train.csv: id, url, landmark_id. https://s3.amazonaws.com/google-landmark/metadata/train.csv
#train_clean.csv: landmark_id, images (' ' separated list of image ids from kaggle retrieval challenge winner) . https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
#train_label_to_category.csv: landmark_id, category (Wikimedia URL to the class definition) fields. https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv

url_img = 'https://s3.amazonaws.com/google-landmark/metadata/train.csv'
data = urlopen(Request(url_img))
df_img_url_landmark = pd.read_table(data, sep=',')

url_landmark_img_cleaned = 'https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv'
data = urlopen(Request(url_landmark_img_cleaned))
df_landmark_img_cleaned = pd.read_table(data, sep=',')

url_landmark = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'
data = urlopen(Request(url_landmark))
df_landmark = pd.read_table(data, sep=',')

#From the cleaned subset, remove images of landmarks which have less than 150 images under the same landmark_id
df_landmark_img_cleaned['images']=list(df_landmark_img_cleaned['images'].str.split())
df_landmark_img_cleaned = df_landmark_img_cleaned[df_landmark_img_cleaned['images'].apply(lambda x: len(x) >= landmark_pop_filter)]

#Extract category name from wikimedia url
df_landmark['category'] = df_landmark['category'].str.split(':',3).str[2]

#Remove categories with unknown characters %,?
df_landmark = df_landmark[~df_landmark['category'].str.contains('%|\?')]

#Remove categories with length<=2
df_landmark = df_landmark[~(df_landmark['category'].str.len()<=cat_char_limit)]

#Remove repeating categories
df_landmark = df_landmark[~df_landmark['category'].duplicated(keep=False)]

#Combine imgs and landmark category info
df_img_landmark_cat_cleaned = pd.merge(df_landmark_img_cleaned, df_landmark, how='inner', on='landmark_id')

df_img_landmark_cat_cleaned = df_img_landmark_cat_cleaned.explode('images').reset_index(drop=True).rename(columns={'images':'id'})

#Select 10 random categories to only include in validation, 10 only in test set and 200 to spread accross train/test/validation
random.seed(33)
train_cats, valid_test_cats = train_test_split(df_img_landmark_cat_cleaned['category'].unique(), train_size =(train_cat_num/df_img_landmark_cat_cleaned['category'].nunique()), random_state=33)
valid_cats = random.sample(list(valid_test_cats), valid_test_cat_num)
test_cats = random.sample(list(valid_test_cats), valid_test_cat_num)

df_valid_all = df_img_landmark_cat_cleaned[df_img_landmark_cat_cleaned['category'].isin(valid_cats)]
df_test_all = df_img_landmark_cat_cleaned[df_img_landmark_cat_cleaned['category'].isin(test_cats)]
df_train_all = df_img_landmark_cat_cleaned[df_img_landmark_cat_cleaned['category'].isin(train_cats)]

#Select 25 images per category to have a balanced representation in training set and 5 per category for test&validation sets
df_valid = df_valid_all.groupby('category').apply(lambda x: x.sample(n=valid_test_per_cat))
df_test = df_test_all.groupby('category').apply(lambda x: x.sample(n=valid_test_per_cat))
df_train = df_train_all.groupby('category').apply(lambda x: x.sample(n=train_per_cat)).reset_index(drop=True)

#Add 5 images per train category into test and validation sets
df_valid = pd.concat([ df_valid, df_train_all.groupby('category').apply(lambda x: x.sample(n=valid_test_per_cat)) ]).reset_index(drop=True)
df_test = pd.concat([ df_test, df_train_all.groupby('category').apply(lambda x: x.sample(n=valid_test_per_cat)) ]).reset_index(drop=True)

df_train.to_csv('./train_ids.csv', index=False)
df_valid.to_csv('./valid_ids.csv', index=False)
df_test.to_csv('./test_ids.csv', index=False)

#Get images that are in the selected subset
train_imgs = list(df_train['id'])
valid_imgs = list(df_valid['id'])
test_imgs = list(df_test['id'])

total_train = 0
total_valid = 0
total_test = 0

for i in range(tar_file_nums):

  tar_num = str(i).zfill(3)
  print(tar_num)

  saved_train = 0
  saved_valid = 0
  saved_test = 0

  url_tar = 'https://s3.amazonaws.com/google-landmark/train/images_' + tar_num + '.tar'
  
  with urllib.request.urlopen(url_tar) as tar_file:
    with tarfile.open(fileobj=tar_file, mode='r|*') as tar:
      tar.extractall(temp_path)

  for dir_paths, dirs, files in os.walk(temp_path):
    for file in files:
      if file.lower().endswith('jpg'):
        source_path = os.path.join(dir_paths, file)

        if file.lower().split('.',2)[0] in train_imgs:
          shutil.copy2(source_path, train_save_path)
          saved_train += 1
          total_train += 1

        if file.lower().split('.',2)[0] in valid_imgs:
          shutil.copy2(source_path, valid_save_path)
          saved_valid += 1
          total_valid += 1

        if file.lower().split('.',2)[0] in test_imgs:
          shutil.copy2(source_path, test_save_path)
          saved_test += 1
          total_test += 1

  print('Train: {} saved from tar file {}, total images: {}'.format(saved_train, tar_num, total_train) )
  print('Validation: {} saved from tar file {}, total images: {}'.format(saved_valid, tar_num, total_valid) )
  print('Test: {} saved from tar file {}, total images: {}'.format(saved_test, tar_num, total_test) )
          
  shutil.rmtree(temp_path)
