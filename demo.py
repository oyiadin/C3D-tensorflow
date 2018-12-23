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

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import cv2
from PIL import Image

labels = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'BasketballDunk', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10 , 'Batch size.')
FLAGS = flags.FLAGS

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

def demo_main_loop(cap):
  model_name = "./sports1m_finetuning_ucf101.model"
  crop_size = 112
  np_mean = np.load('crop_mean.npy').reshape([16, crop_size, crop_size, 3])

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

  #############################################################################
  #############################################################################
  #############################################################################

  # main loop
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

      cv2.imshow('frame', cv2.resize(frame, (256, 144)))
      if cv2.waitKey(15) & 0xFF == ord('q'):
        print('goodbye')
        return

    img_datas = np.array(img_datas)
    # to label the last 16 frames captured by opencv2 using our model
    zeros = np.tile(np.zeros_like(img_datas[np.newaxis, :]).T, 20).T
    print(zeros.shape, np.array(img_datas).shape)
    test_images = zeros
    test_images[0] = img_datas
    # convert to (20, 16, 112, 112, 3)

    start_time = time.time()
    score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )[0]
    top1_predicted_label = np.argmax(score)
    print(labels[top1_predicted_label])
    print(' (It took %ds to get this result...)' % int(time.time() - start_time))

def main(_):
  #try:
  cap = cv2.VideoCapture(0)
  demo_main_loop(cap)
  #except KeyboardInterrupt:
  #  print('goodbye')
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  tf.app.run()
