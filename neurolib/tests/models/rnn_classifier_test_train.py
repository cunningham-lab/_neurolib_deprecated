# Copyright 2018 Daniel Hernandez Diaz, Columbia University
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
#
# ==============================================================================
import os
os.environ['PATH'] += ':/usr/local/bin'
import unittest

import numpy as np
import tensorflow as tf

from neurolib.models.rnn_sequence_predictor import RNNClassifier

# pylint: disable=bad-indentation, no-member, protected-access

def generateEchoData(num_labels, length, echo_step):
  """
  """
  p = num_labels*[1/num_labels]
  x = np.array(np.random.choice(num_labels, length, p=p))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0

  return x, y

class RNNClassifierTrainTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  def test_build(self):
    """
    Test build
    """
    num_labels = 4
    max_steps = 25
    echo_step = 3
    X, Y = generateEchoData(num_labels, 10000, echo_step)
    X = np.reshape(X, [-1, max_steps, 1])
    Y = np.reshape(Y, [-1, max_steps, 1])
    train_dataset = {'train_iseq' : X[:300],
                     'train_labels' : Y[:300]}
    
    model = RNNClassifier(num_labels,
                          input_dim=1,
                          hidden_dim=4,
                          batch_size=1,
                          max_steps=max_steps)
    model.build()
    model.train(train_dataset)
    
    # Test on validation data
    dataset = {'iseq' : X[390:]}
    Ypred = model.sample(dataset)
    print("Preds:", list(zip(np.squeeze(X[390], axis=1)[:-echo_step],
          np.argmax(np.exp(Ypred[0].T)/np.sum(np.exp(Ypred[0]), axis=1),
                    axis=0)[echo_step:])))
    
if __name__ == '__main__':
  unittest.main(failfast=True) 