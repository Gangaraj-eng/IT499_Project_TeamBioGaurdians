from random import shuffle
import cv2
import tensorflow as tf
import sys
import pandas as pd
import os
from audioPreprocessing import extract_audio_feature
import numpy as np

def _float_array_feature(array):
    tensor = tf.convert_to_tensor(array)
    value =  tf.io.serialize_tensor(tensor).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
# required output dimensions for the image
img_dimensions = (224,224)
rootAudioPath = r'D:\BiometricProject\Mini_Data\Audio\Dev'
rootImgPath = r'D:\BiometricProject\Mini_Data\Images\Dev'

def load_imag(imgPath):
     # read the image and resize into a fixed size
     
     img = cv2.imread(os.path.join(rootImgPath, imgPath))
     
     if img is None:
       return None
     
     # perform resizing 
     img = cv2.resize(img, img_dimensions)
     
     # rescaling to normalize
     img = img/255.0
     return img
   

def createDataRecord_static1(outfileName, img_addrs, audio_addrs, labels):
    
    writer = tf.io.TFRecordWriter(outfileName)
    
    for i in range(len(img_addrs)):
      
      if i%1000 == 0:
        print('Created {}/{} records'.format(i, len(img_addrs)))
        sys.stdout.flush()
        
      img = load_imag(img_addrs[i])
      
      # load audio file image
      audio = extract_audio_feature(os.path.join(rootAudioPath,audio_addrs[i]), 'mfcc', n_mfcc=20)
      audio = np.array(audio, dtype=np.float64)
      label = labels[i]
      
      if img is None:
        continue
      feature={
        'image_raw': _float_array_feature(img),
        'audio_raw': _float_array_feature(audio),
        'label': _int64_feature(label)
      }
      
      example = tf.train.Example(features = tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
      
    writer.close()
    sys.stdout.flush()
    

def createDataRecord_static2(outfileName, img1_addrs, img2_addrs, audio_addrs, labels):
    writer = tf.io.TFRecordWriter(outfileName)
    
    for i in range(len(audio_addrs)):
      
      if i%1000 == 0:
        print('Created {}/{} records'.format(i, len(audio_addrs)))
        sys.stdout.flush()
        
      img1 = load_imag(img1_addrs[i])
      img2 = load_imag(img2_addrs[i])
      
      # load audio file image
      audio = extract_audio_feature(os.path.join(rootAudioPath,audio_addrs[i]), 'mfcc', n_mfcc=20)
      audio = np.array(audio, dtype=np.float64)
      label = labels[i]
      
      if img1 is None or img2 is None:
        continue
      feature={
        'image1_raw': _float_array_feature(img1),
        'image2_raw': _float_array_feature(img2),
        'audio_raw': _float_array_feature(audio),
        'label': _int64_feature(label)
      }
      
      example = tf.train.Example(features = tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
      
    writer.close()
    sys.stdout.flush()


# static architecture 1
# labelledDataset = pd.read_csv('labelledData.csv')


# imgFilePaths = labelledDataset['f'].values
# audFilePaths = labelledDataset['v'].values
# labels = labelledDataset['y'].values
# createDataRecord_static1('train.tfrecords',imgFilePaths, audFilePaths, labels)



# static architecture 2
labelledDataset = pd.read_csv('../labelledDatasetStatic2.csv')
img1FilePaths = labelledDataset['face1'].values[0:10]
img2FilePaths = labelledDataset['face2'].values[0:10]
audioFilePaths = labelledDataset['voice'].values[0:10]
labels = labelledDataset['label']
createDataRecord_static2('train2.tfrecords',img1FilePaths, img2FilePaths, audioFilePaths, labels)