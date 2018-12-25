# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import random
import cv2
from PIL import Image

labels = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'BasketballDunk', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10 , 'Batch size.')
FLAGS = flags.FLAGS

model_name = "./sports1m_finetuning_ucf101.model"
crop_size = 112
np_mean = np.load('crop_mean.npy').reshape([16, crop_size, crop_size, 3])


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var


def init_ops():
    # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  logits = []
  for gpu_index in range(0, gpu_num):
    with tf.device('/gpu:%d' % gpu_index):
      logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
      logits.append(logit)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, model_name)
  # And then after everything is built, start the training loop.
  return sess, images_placeholder, norm_score


def demo_main_loop(cap):
  sess, X, norm_score = init_ops()

  #############################################################################
  #############################################################################
  #############################################################################

  # main loop
  top1_predicted_label = 'None'
  while True:
    img_datas = []

    # reads frames from the camera and accumulate them until we have 16 frames
    for step in range(2*16):
      ret, frame = cap.read()
      while not ret:
        print('cannot read correctly, retrying')
        ret, frame = cap.read()

      if not step % 2:
        # preprocessing the frames so that it can suits for our pretrained model
        img = Image.fromarray(frame.astype(np.uint8))
        # assert img.width > img.height:
        scale = float(crop_size) / float(img.height)
        img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :] - np_mean[len(img_datas)]
        img_datas.append(img)

      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(frame, top1_predicted_label, (10, 100), font, 3, (255,255,255),5, cv2.LINE_AA, bottomLeftOrigin=False)

      cv2.imshow('frame', cv2.resize(frame, (384, 216)))
      if cv2.waitKey(15) & 0xFF == ord('q'):
        print('goodbye')
        return

    img_datas = np.array(img_datas)
    # to label the last 16 frames captured by opencv2 using our model
    zeros = np.tile(np.zeros_like(img_datas[np.newaxis, :]).T, 20).T
    # print(zeros.shape, np.array(img_datas).shape)
    test_images = zeros
    test_images[0] = img_datas
    # convert to (20, 16, 112, 112, 3)

    start_time = time.time()
    score = norm_score.eval(
            session=sess,
            feed_dict={X: test_images}
            )[0]
    top1_predicted_label = labels[np.argmax(score)]
    print('the top 1 predicted label is:', top1_predicted_label)
    print('  (It took %ds to get this result...)\n' % int(time.time() - start_time))


def gen_list(inf, outf):
  with open(inf) as f:
    f = list(f)
    selected_lines = random.sample(f, 20)
    with open(outf, 'w') as ff:
      ff.write(''.join(selected_lines))
      ff.close()
  print('list generated')


def gen_demo_video(inf, outf):
  print('loading list file..')
  lines = [i.strip().split() for i in list(open(inf))]
  assert len(lines) == 20
  dirs = [i[0] for i in lines]
  true_labels = [i[1] for i in lines]

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  out = cv2.VideoWriter(outf, fourcc, 16.0, (512, 288))

  print('init and restore the model..')
  sess, X, norm_score = init_ops()
  all_data = []
  all_original = []
  for i in range(len(dirs)):
    print('preprocessing', dirs[i])
    preprocessed_frames = []
    original_frames = input_data.get_frames_data(dirs[i])[0][:16]
    for frame in original_frames:
      # preprocessing the frames so that it can suits for our pretrained model
      img = Image.fromarray(frame.astype(np.uint8))
      # assert img.width > img.height:
      scale = float(crop_size) / float(img.height)
      img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
      crop_x = int((img.shape[0] - crop_size)/2)
      crop_y = int((img.shape[1] - crop_size)/2)
      img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :] - np_mean[len(preprocessed_frames)]
      preprocessed_frames.append(img)
    # print(np.array(preprocessed_frames).shape)
    all_data.append(np.array(preprocessed_frames))
    all_original.append(np.array(original_frames))
  # print(np.array(all_data).shape)

  print('all data have already been preprocessed')
  print('predicting..')
  start_time = time.time()
  scores = norm_score.eval(
    session=sess,
    feed_dict={X: np.array(all_data)}
  )
  pred_labels = []
  for i in scores:
    pred_labels.append(np.argmax(i))
    print('{:>30} : true_label = {} ; pred_label = {}'.format(
      dirs[len(pred_labels)-1].split(os.sep)[-1],
      true_labels[len(pred_labels)-1],
      pred_labels[-1]))
  print('predicting took me {}s'.format(int(time.time() - start_time)))

  print('saving to video...')
  for n, video in enumerate(all_original):
    print('saving: [ {:0>2} / {:0>2} ]'.format(n+1, len(all_original)))
    for frame in video:
      font = cv2.FONT_HERSHEY_SIMPLEX
      color = (0, 255, 127) if pred_labels[n] == true_labels[n] else (255, 0, 0)
      cv2.putText(frame, labels[pred_labels[n]], (10, 40),
        font, 1, color, 2, cv2.LINE_AA, bottomLeftOrigin=False)
      resized = cv2.resize(frame, (512, 288))
      out.write(resized)
  
  print('saved, goodbye!')
  out.release()


def main(_):
  if len(sys.argv) >= 2:
    if sys.argv[1].lower() == 'gen_list':
      gen_list(sys.argv[2], sys.argv[3])
    elif sys.argv[1].lower() == 'demo_video':
      gen_demo_video(sys.argv[2], sys.argv[3])
    elif sys.argv[1].lower() == 'help':
      print('* python3 demo.py gen_list in_file out_file')
      print('* python3 demo.py demo_video in_file out_file')
      print('* python3 demo.py')
      print('  (this will capture images from your camera)')
    else:
      print('unknown option, run `python3 demo.py help`')
  else:
    try:
      cap = cv2.VideoCapture(0)
      demo_main_loop(cap)
    except KeyboardInterrupt:
      print('goodbye')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  tf.app.run()
