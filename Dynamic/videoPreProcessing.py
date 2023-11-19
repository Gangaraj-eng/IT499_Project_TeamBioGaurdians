import tensorflow as tf
import cv2
import random
import numpy as np


# function that resize a frame into desired output size
def format_frames(frame, output_size = (224,224)):
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  # resize with padding inorder to obtain the desired shape
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

# function to extract the frames form a given video file path
# n_frames - number of output frames to be extracted
# output_size - dimensions of each frame
# frame_stpe - gap between two collected frames
def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
      start = 0
    else:
      max_start = video_length - need_length
      start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
      for _ in range(frame_step):
        ret, frame = src.read()
      if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
      else:
        result.append(np.zeros_like(result[0]))
    src.release()
    
    result = np.array(result)[..., [2, 1, 0]]

    return result
