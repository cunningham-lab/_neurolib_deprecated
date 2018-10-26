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

import tensorflow as tf

from neurolib.models.rnn_sequence_predictor import RNNClassifier

# pylint: disable=bad-indentation, no-member, protected-access
        
class RNNClassifierBuildTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  def test_init(self):
    """
    Test initialization
    """
    print("\nTest 0: Regression initialization")
    RNNClassifier(2, input_dim=1, hidden_dim=3)
    
  def test_build(self):
    """
    Test build
    """
    model = RNNClassifier(2, input_dim=1, hidden_dim=3)
    model.build()


if __name__ == '__main__':
  unittest.main(failfast=True) 