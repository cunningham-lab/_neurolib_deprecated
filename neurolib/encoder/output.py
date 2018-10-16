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
import pydot
import tensorflow as tf

from neurolib.encoder.anode import ANode

# pylint: disable=bad-indentation, no-member, protected-access

class OutputNode(ANode):
  """
  An OutputNode represents a sink of information in the Model graph (MG).
  OutputNodes are mainly useful for model bending (training). A cost function
  for instance depends only on the inputs of OutputNodes.
    
  OutputNodes have no outputs, that is, information is "destroyed" at the
  OutputNode. Assignment to self.num_declared_outputs is therefore forbidden.
  
  OutputNodes have a single input assigned to islot = 0. Every output node maps
  to a single tensor, its input. These tensors can then be invoked by the
  OutputNode name.
  
  Class attributes:
    num_expected_inputs = 1
    num_expected_outputs = 0
  """
  num_expected_inputs = 1
  num_expected_outputs = 0
  
  def __init__(self, label, name=None):
    """
    Initialize the OutputNode
    
    Args:
      label (int): A unique integer identifier for the node.
      
      name (str): A unique name for this node.
    """
    self.name = "Out_" + str(label) if name is None else name
    
    super(OutputNode, self).__init__(label)
    
    # Add visualization
    self.vis = pydot.Node(self.name)

  @ANode.num_declared_inputs.setter
  def num_declared_inputs(self, value):
    """
    Setter for num_declared_inputs
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_declared_inputs of OutputNodes"
                           "must be either 0 or 1")
    self._num_declared_inputs = value

  @ANode.num_declared_outputs.setter
  def num_declared_outputs(self):
    """
    Setter for num_declared_outputs. 
    
    Raises an error, num_declared_outputs is fixed to 0.
    """
    raise AttributeError("Assignment to attribute num_declared_outputs "
                         "of OutputNodes is disallowed. num_declared_outputs "
                         "is fixed to 0 for an OutputNode")
    
  def _build(self):
    """
    Build the OutputNode.
    
    Specifically, rename the input tensor to the name of the node so that it can
    be easily accessed.
    
    NOTE: The _islot_to_itensor attribute of this node has been updated by the
    Builder object during processing of this OutputNode's parent by the
    Builder build algorithm
    """
    print("\nOutput:", self.name,
          "\nself._islot_to_itensor[0]", type(self._islot_to_itensor[0]) )
    self._islot_to_itensor[0] = tf.identity(self._islot_to_itensor[0],
                                            name=self.name)
    
    self._is_built = True
    