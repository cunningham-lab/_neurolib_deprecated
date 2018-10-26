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

from neurolib.models.vae import VariationalAutoEncoder 

# pylint: disable=bad-indentation, no-member, protected-access


class VAETestBuild(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  def test_init(self):
    """
    """
    VariationalAutoEncoder(latent_dim=3,
                           output_dim=10,
                           batch_size=1)
    
  def test_build(self):
    """
    """
    dc = VariationalAutoEncoder(latent_dim=3, output_dim=10, batch_size=1)
    dc.build()


if __name__ == '__main__':
  unittest.main(failfast=True)