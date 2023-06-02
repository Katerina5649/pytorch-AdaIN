"""
  Generate landmark-style pairs for training, validation and test sets.
"""

import pandas as pd
import os
import shutil
import numpy as np
from tqdm import tqdm

# optional
os.chdir('Style_imgs')

# I/O 
STYLE_FOLDER = './imgs/' # style images folder
LANDMARK_FOLDER = '../../input/EPFL_landmark/' # landmark images folder
STYLES_METADATA_FILE = STYLE_FOLDER + 'all_data_info.csv' # metadata file for style images
OUTPUT_FOLDER = '../../input/EPFL_styles/' # output folder for the pairs

# Pairing generation parameters
n_baseline_imgs = {'all': 10, 'val': 5, 'test': 5} # number of stylized images paired with each landmark image in the split
n_rnd_imgs = {'train': 10, 'val': 5, 'test': 5} # number of stylized images paired randomly with each landmark image in the split


#
# Import metadata
#

# import style images metadata
metadata = pd.read_csv(STYLES_METADATA_FILE, sep=',', usecols=['new_filename', 'artist', 'genre', 'style', 'title'])
metadata = metadata.rename(columns={'new_filename': 'style_img_path'})

# define the baselines for each split
baselines = {
    'all': metadata.sample(n=n_baseline_imgs['all']),
    'val': metadata.sample(n=n_baseline_imgs['val']),
    'test': metadata.sample(n=n_baseline_imgs['test'])
}

# remove the baselines from the metadata to avoid duplicates
metadata = metadata.drop(baselines['all'].index)
metadata = metadata.drop(baselines['val'].index)
metadata = metadata.drop(baselines['test'].index)

# import landmark images metadata
def import_landmark_metadata(split):
  landmarks = pd.read_csv(LANDMARK_FOLDER + split + '_ids.csv', sep=',')
  landmarks['landmark_img_path'] = landmarks['id'].apply(lambda x: str(x) + '.jpg')
  return landmarks

landmark_train = import_landmark_metadata('train')
landmark_val = import_landmark_metadata('valid')
landmark_test = import_landmark_metadata('test')

# define landmarks intersection. These landmark images are present in different
# splits and therefore the baseline['all'] cannot be applied to them, otherwise it will
# create duplicates
train_val_inter = np.intersect1d(landmark_val['id'].unique(), landmark_train['id'].unique())
test_val_inter = np.intersect1d(landmark_val['id'].unique(), landmark_test['id'].unique())
test_train_inter = np.intersect1d(landmark_train['id'].unique(), landmark_test['id'].unique())

landmark_intersect = {
    'val': np.union1d(test_val_inter, test_train_inter),
    'test': np.union1d(train_val_inter, test_val_inter)
}


# create the landmark - style pairs given a landmark image
def generate_pairs_one_landmark(landmark_img, split, samples, baseline_type, n_rnd_imgs=0):
  """

    landmark_img: row of the landmark dataframe
    baseline_type: type of baseline to use (all, val, test)
    split: name of the split (train, val, test)
    samples: dataframe containing the baseline images or None if to use random images
    n_rnd_imgs: number of random images to use. Set a value if the baseline is not provided in `samples`
 """
  global metadata
  if samples is None:
    # remove the samples from the metadata to avoid duplicates
    samples = metadata.sample(n=n_rnd_imgs)
    metadata = metadata.drop(samples.index)

  # If the landmark is in multiple sets, don't use the baseline but use random style images
  # the random style images needs to have the same size as the baseline for coherence.
  # We have to check if the baseline type is all, as the baseline for val and test 
  # are only applied to these respective splits
  if baseline_type == 'all' and split in ['val', 'test'] and landmark_img['id'] in landmark_intersect[split]:    
    samples = metadata.sample(n=n_baseline_imgs['all'])
    metadata = metadata.drop(samples.index)

  # create the pairs
  new_rows = pd.DataFrame(samples.reset_index(drop=True))
  new_rows[['landmark_id', 'category', 'landmark_img_path']] = landmark_img[['landmark_id', 'category', 'landmark_img_path']]
  return new_rows


def generate_pairs_all_landmarks(landmark_df, split):
  """
    Create a dataframe with the baseline and random landmark-style pairs for the given split
    Note that the baseline is the same for every split, and val and test have a second smalle baseline

    landmark_df: dataframe containing the landmark images
    baseline_df: dataframe containing the baseline images
    split: name of the split (train, val, test)
  """
  assert(split in ['train', 'val', 'test'])
  
  landmark_style_pairs_baseline = pd.concat(landmark_df.apply(generate_pairs_one_landmark, args=(split, baselines['all'], 'all'), axis=1).to_list())
  landmark_style_pairs_baseline['type'] = 'baseline'

  assert(landmark_style_pairs_baseline.shape[0] == n_baseline_imgs['all'] * landmark_df.shape[0])

  landmark_style_pairs_random = pd.concat(landmark_df.apply(generate_pairs_one_landmark, args=(split, None, None, n_rnd_imgs[split]), axis=1).to_list())
  landmark_style_pairs_random['type'] = 'random'

  assert(landmark_style_pairs_random.shape[0] == n_rnd_imgs[split] * landmark_df.shape[0])

  landmark_style_pairs = pd.concat([landmark_style_pairs_baseline.reset_index(drop=True), landmark_style_pairs_random.reset_index(drop=True)])

  # add the split baseline values if needed
  if split in ['val', 'test']:  
    landmark_style_pairs_baseline_split = pd.concat(landmark_df.apply(generate_pairs_one_landmark, args=(split, baselines[split], split), axis=1).to_list())
    landmark_style_pairs_baseline_split['type'] = 'baseline' + split

    assert(landmark_style_pairs_baseline_split.shape[0] == n_baseline_imgs[split] * landmark_df.shape[0])

    landmark_style_pairs = pd.concat([landmark_style_pairs.reset_index(drop=True), landmark_style_pairs_baseline_split.reset_index(drop=True)])
  
  # add split column
  landmark_style_pairs['split'] = split

  return landmark_style_pairs

# create landmark-style pairs for each set
print("Compute pairs for train...")
landmark_style_pairs_train = generate_pairs_all_landmarks(landmark_train, 'train')
print("Compute pairs for val...")
landmark_style_pairs_val = generate_pairs_all_landmarks(landmark_val, 'val')
print("Compute pairs for test...")
landmark_style_pairs_test = generate_pairs_all_landmarks(landmark_test, 'test')


#
# Assertions before exporting and moving the data
#

# check that the generates datasets have the correct dimensions
assert(landmark_style_pairs_train.shape[0] == landmark_train.shape[0] * (n_baseline_imgs['all'] + n_rnd_imgs['train']))
assert(landmark_style_pairs_val.shape[0] == landmark_val.shape[0] * (n_baseline_imgs['all'] + n_rnd_imgs['val'] + n_baseline_imgs['val']))
assert(landmark_style_pairs_test.shape[0] == landmark_test.shape[0] * (n_baseline_imgs['all'] + n_rnd_imgs['test'] + n_baseline_imgs['test']))

# assertions check that only the baseline is present in each split
inter_train_val = np.intersect1d(landmark_style_pairs_train['style_img_path'].unique(), landmark_style_pairs_val['style_img_path'].unique())
inter_train_test = np.intersect1d(landmark_style_pairs_train['style_img_path'].unique(), landmark_style_pairs_test['style_img_path'].unique())
inter_val_test = np.intersect1d(landmark_style_pairs_val['style_img_path'].unique(), landmark_style_pairs_test['style_img_path'].unique())

baseline_paths = baselines['all']['style_img_path'].to_numpy()

assert(np.intersect1d(inter_train_val, baseline_paths).shape[0] == len(baseline_paths))
assert(np.intersect1d(inter_train_test, baseline_paths).shape[0] == len(baseline_paths))
assert(np.intersect1d(inter_val_test, baseline_paths).shape[0] == len(baseline_paths))

# check that only baseline images are repeated
def check_baseline_repetition(landmark_style_pairs, split):
    theoritical_res = baselines['all']['style_img_path'].to_numpy()

    if split in ['val', 'test']:
        theoritical_res = np.concatenate((theoritical_res, baselines[split]['style_img_path'].to_numpy()))

    counts = landmark_style_pairs['style_img_path'].value_counts()
    empirical_res = counts[counts > 1].index.to_numpy()
    
    return np.intersect1d(theoritical_res, empirical_res).shape[0] == len(theoritical_res)


assert(check_baseline_repetition(landmark_style_pairs_train, 'train'))
assert(check_baseline_repetition(landmark_style_pairs_val, 'val'))
assert(check_baseline_repetition(landmark_style_pairs_test, 'test'))


#
# Export and move the data
#
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

landmark_style_pairs_train.to_csv(OUTPUT_FOLDER + 'landmark_style_pairs_train.csv', index=False, sep=',')
print("Exported ", OUTPUT_FOLDER + 'landmark_style_pairs_train.csv')
landmark_style_pairs_val.to_csv(OUTPUT_FOLDER + 'landmark_style_pairs_val.csv', index=False, sep=',')
print("Exported ", OUTPUT_FOLDER + 'landmark_style_pairs_val.csv')
landmark_style_pairs_test.to_csv(OUTPUT_FOLDER + 'landmark_style_pairs_test.csv', index=False, sep=',')
print("Exported ", OUTPUT_FOLDER + 'landmark_style_pairs_test.csv')

# move all files of the output folder
def move_images(landmark_style_pairs, split):  
  output_folder_path = OUTPUT_FOLDER + split
  print(f"Moving {split} images to: {output_folder_path}")

  # remove if exists, then recreate  
  if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)
  os.makedirs(output_folder_path)
  
  for style_img_path in tqdm(landmark_style_pairs['style_img_path'].unique()):
    if not os.path.exists(STYLE_FOLDER + 'img/' + style_img_path):
      print("file not found: ", STYLE_FOLDER + style_img_path)
    else:
      shutil.copyfile(STYLE_FOLDER + 'img/' + style_img_path, output_folder_path + '/' + style_img_path)
      

move_images(landmark_style_pairs_train, 'train')
move_images(landmark_style_pairs_val, 'val')
move_images(landmark_style_pairs_test, 'test')