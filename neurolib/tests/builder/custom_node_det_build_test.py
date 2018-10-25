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
import unittest
import tensorflow as tf

from neurolib.builders.static_builder import StaticBuilder

# pylint: disable=bad-indentation, no-member, protected-access

class CustomEncoderBuilderBasicTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(False, "Skipping")
  def test_init(self):
    """
    Create a CustomNode
    """
    tf.reset_default_graph()
    builder = StaticBuilder()
    builder.createCustomNode(1, 1, name="Custom")
  
  @unittest.skipIf(False, "Skipping")
  def test_add_encoder0(self):
    """
    Test commit
    """
    tf.reset_default_graph()
    builder = StaticBuilder("MyModel")

    builder.addInput(10)
    cust = builder.createCustomNode(1, 1, name="Custom")
    cust_in1 = cust.addInner(3)
    cust_in2 = cust.addInner(4)
    cust.addDirectedLink(cust_in1, cust_in2)
    cust.commit()
    builder.addOutput()
      
  @unittest.skipIf(False, "Skipping")
  def test_add_encoder1(self):
    """
    Test build
    """
    tf.reset_default_graph()
    builder = StaticBuilder("MyModel")

    cust = builder.createCustomNode(1, 1, name="Custom")
    cust_in1 = cust.addInner(3)
    cust_in2 = cust.addInner(4)
    cust.addDirectedLink(cust_in1, cust_in2)
    cust.commit()
     
    in1 = builder.addInput(10)
    o1 = builder.addOutput()
    builder.addDirectedLink(in1, cust)
    builder.addDirectedLink(cust, o1)
     
    builder.build()
    
    
if __name__ == "__main__":
  unittest.main(failfast=True)