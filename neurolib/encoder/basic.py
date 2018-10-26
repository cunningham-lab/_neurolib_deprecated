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
from abc import abstractmethod

import tensorflow as tf

import pydot

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode

    
# pylint: disable=bad-indentation, no-member

class InnerNode(ANode):
  """
  Abstract class for interior nodes
  
  An InnerNode is an Anode that resides in the interior of the model graph. An
  InnerNode performs an operation on its inputs producing its outputs.
  Alternatively, an InnerNode can be defined as any node that is not an
  OutputNode nor an InputNode. InnerNodes have num_inputs > 0 and num_outputs >
  0. Its outputs can be deterministic, as in the DeterministicNNNode, or
  stochastic, as in the NormalTriLNode.
  
  The InnerNode should implement `__call__` and `_build`.
  """
  def __init__(self,
               builder,
               is_sequence=False):
    """
    Initialize the InnerNode
    
    Args:
      label (int): A unique integer identifier for the InnerNode
    """
    self.builder = builder
    self.label = builder.num_nodes
    builder.num_nodes += 1

    self.batch_size = builder.batch_size
    self.max_steps = builder.max_steps if hasattr(builder, 'max_steps') else None
    self.is_sequence = is_sequence

    super(InnerNode, self).__init__()
                  
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    Call the node transformation on inputs, return the outputs.
    """
    if inputs is not None:
      print("inputs", inputs)
      try:
        islot_to_itensor = dict(enumerate(inputs))
      except TypeError:
        islot_to_itensor = dict([0, inputs])
    return self._build(islot_to_itensor) 
  
  @abstractmethod
  def _build(self, islot_to_itensor):
    """
    """
    raise NotImplementedError("Please implement me.")

 
class CopyNode(InnerNode):
  """
  A utility node that copies its input to its output.
  """
  num_expected_outputs = 1
  num_expected_inputs = 1
  
  def __init__(self,
               builder,
               name=None):
    """
    Initialize the CopyNode
    """
    super(CopyNode, self).__init__(builder)
    self.name = "Copy_" + str(self.label) if name is None else name
    
  def _build(self, islot_to_itensor=None):
    """
    Build the CopyNode
    """
    if islot_to_itensor is None:
      islot_to_itensor = self._islot_to_itensor

    output = _input = islot_to_itensor[0] # make sure the inputs are ordered
    output_name = self.name + '_out'
    
    self._oslot_to_otensor[0] = tf.identity(output, output_name)
    
    
if __name__ == '__main__':
  print(dist_dict)