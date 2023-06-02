import numpy as np
import pandas as pd
import os
import urllib
import shutil
import io
import random

train_path = './EPFL_stylized/train/'
valid_path = './EPFL_stylized/val/'
test_path = './EPFL_stylized/test/'

train_names =[]
for dir_paths, dirs, files in os.walk(train_path):
  for file in files:
    if file.lower().endswith('jpg'):
      train_names.append(file.lower().split('.',2)[0])
      
valid_names =[]
for dir_paths, dirs, files in os.walk(valid_path):
  for file in files:
    if file.lower().endswith('jpg'):
      valid_names.append(file.lower().split('.',2)[0])
      
test_names =[]
for dir_paths, dirs, files in os.walk(test_path):
  for file in files:
    if file.lower().endswith('jpg'):
      test_names.append(file.lower().split('.',2)[0])
      
train_pairs = pd.DataFrame(train_names, columns=['stylized_name'])
train_pairs[['img_id', 'place_holder', 'style_id']]=(train_pairs['stylized_name'].str.split('_',2, expand=True))
train_pairs.drop('place_holder', axis=1, inplace=True)

valid_pairs = pd.DataFrame(valid_names, columns=['stylized_name'])
valid_pairs[['img_id', 'place_holder', 'style_id']]=(valid_pairs['stylized_name'].str.split('_',2, expand=True))
valid_pairs.drop('place_holder', axis=1, inplace=True)

test_pairs = pd.DataFrame(test_names, columns=['stylized_name'])
test_pairs[['img_id', 'place_holder', 'style_id']]=(test_pairs['stylized_name'].str.split('_',2, expand=True))
test_pairs.drop('place_holder', axis=1, inplace=True)

train_cats = pd.read_csv('../data_extraction/get_landmark_imgs/train_ids.csv')
valid_cats = pd.read_csv('../data_extraction/get_landmark_imgs/valid_ids.csv')
test_cats = pd.read_csv('../data_extraction/get_landmark_imgs/test_ids.csv')

train_pairs = pd.merge(train_pairs, train_cats, how='inner', right_on='id', left_on='img_id')
train_pairs.drop('id', axis=1, inplace=True)

valid_pairs = pd.merge(valid_pairs, valid_cats, how='inner', right_on='id', left_on='img_id')
valid_pairs.drop('id', axis=1, inplace=True)

test_pairs = pd.merge(test_pairs, test_cats, how='inner', right_on='id', left_on='img_id')
test_pairs.drop('id', axis=1, inplace=True)

def get_other_imgs(img_id, landmark_id, num=5, cats):
  all_imgs = [x for x in cats[cats['landmark_id']==landmark_id]['id'] if x != img_id]
  random_imgs = random.sample(all_imgs, k=num)
  return random_imgs
  
train_pairs['paired_imgs'] = train_pairs.apply(lambda x : get_other_imgs(x['img_id'], x['landmark_id'], 5, train_cats), axis=1)
valid_pairs['paired_imgs'] = valid_pairs.apply(lambda x : get_other_imgs(x['img_id'], x['landmark_id'], 2, valid_cats), axis=1)
test_pairs['paired_imgs'] = test_pairs.apply(lambda x : get_other_imgs(x['img_id'], x['landmark_id'], 2, test_cats), axis=1)

train_pairs = train_pairs.explode('paired_imgs').reset_index(drop=True).rename(columns={'paired_imgs':'paired_img'})
valid_pairs = valid_pairs.explode('paired_imgs').reset_index(drop=True).rename(columns={'paired_imgs':'paired_img'})
test_pairs = test_pairs.explode('paired_imgs').reset_index(drop=True).rename(columns={'paired_imgs':'paired_img'})

train_pairs[["stylized_name", "paired_img"]] = train_pairs[["stylized_name", "paired_img"]].apply(lambda x: x + '.jpg')
valid_pairs[["stylized_name", "paired_img"]] = valid_pairs[["stylized_name", "paired_img"]].apply(lambda x: x + '.jpg')
test_pairs[["stylized_name", "paired_img"]] = test_pairs[["stylized_name", "paired_img"]].apply(lambda x: x + '.jpg')

train_pairs.to_csv('./train_pairs.csv', index=False)
valid_pairs.to_csv('./valid_pairs.csv', index=False)
test_pairs.to_csv('./test_pairs.csv', index=False)