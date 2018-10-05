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

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode
    

class InnerNode(ANode):
  """
  An InnerNode is a node that is not an OutputNode nor an InputNode. InnerNodes
  have num_inputs > 0 and num_outputs > 0. Its outputs can be deterministic, as
  in the DeterministicNNNode, or stochastic, as in the NormalTriLNode. 
  
  Some InnerNodes are utility nodes, their purpose being to stitch together
  other InnerNodes into the Model graph (MG). This is the case for instance of
  the MergeConcatNode and the CloneNode.  
  """
  def __init__(self, label):
    """
    Initialize the InnerNode
    
    Args:
      label (int): A unique integer identifier for the node
    """
    super(InnerNode, self).__init__(label)
              
    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')
      

class MergeConcatNode(InnerNode):
  """
  A MergeConcatNode merges its inputs by concatenation to produce a single
  output.
  
  A MergeConcatNode has an arbitrary number of inputs n. n is specified at
  initialization.
  
  Class attributes:
    _requres_builder = False
    num_expected_outputs = 1
  """
  _requires_builder = False
  num_expected_outputs = 1
  
  def __init__(self, label, num_mergers, axis, name=None):
    """
    Initialize a MergeConcatNode
    
    Args:
      label (int): A unique integer identifier for the node
      
      num_mergers (int): The number of inputs to be concatenated
      
      axis (int): The axis to concatenate
      
      name (str): A unique string identifier for this node
    """
    self.label = label
    super(MergeConcatNode, self).__init__(label)
    
    self.num_expected_inputs = self.num_mergers = num_mergers
    self.axis = axis
    self.name = "Concat_" + str(label) if name is None else name
    
    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')
    
  @ANode.num_inputs.setter
  def num_inputs(self, value):
    """
    Setter for self.num_inputs
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of this MergeConcatNode must be "
                           "at most", self.num_expected_inputs)
    self._num_declared_inputs = value

  @ANode.num_outputs.setter
  def num_outputs(self, value):
    """
    Setter for self.num_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of this CloneNode must be "
                           "at most", self.num_expected_outputs)
    self._num_declared_outputs = value
    
  def _update_when_linked_as_node2(self):
    """
    Compute the output shape from the inputs and assign to _oslot_to_shape[0]
    
    NOTE: Before building a node the shapes of all directed edges should be
    specified. That is, the dictionaries _islot_to_shape and _oslot_to_shape
    must be filled for all islots and oslots of the graph nodes. If an oshape
    cannot be provided at node initialization but needs to be inferred from the
    node's inputs, then this method is called by the Builder to do so after
    declaration of the directed link
    """
    if self.num_inputs == self.num_expected_inputs:
      s = 0
      oshape = list(self._islot_to_shape[0])
#       print('oshape concat:', oshape)
      for islot in range(self.num_expected_inputs):
#         print('islot, shape', islot, self._islot_to_shape[islot])
        s += self._islot_to_shape[islot][self.axis]
#         print('s', s)
      oshape[self.axis] = s
#       print('final oshape concat:', oshape)
    
      self._oslot_to_shape[0] = oshape
      
  def _build(self):
    """
    Build the MergeConcatNode
    
    Concatenate the inputs along self.axis and assign to oslot=0
    """
    values = list(self._islot_to_itensor.values())
    self._oslot_to_otensor[0] = tf.concat(values, axis=self.axis)
    
    self._is_built = True

    
class CloneNode(ANode):
  """
  A CloneNode clones a single input to produce many identical outputs.

  A CloneNode has an arbitrary number of outputs n. n must specified at
  initialization.
  
  Class attributes:
    _requres_builder = False
    num_expected_inputs = 1

  """
  _requires_builder = False
  num_expected_inputs = 1
  
  def __init__(self, label, num_clones, name=None):
    """
    Initialize a CloneNode
    
    Args:
      label (int): A unique integer identifier for the node
      
      num_clones (int): The number of outputs to produce
      
      name (str): A unique string identifier for this node
    """
    self.label = label
    super(CloneNode, self).__init__(label)
    
    self.num_clones = self.num_expected_outputs = num_clones
    
    self.name = "Clone_" + str(label) if name is None else name

    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')

  @ANode.num_inputs.setter
  def num_inputs(self, value):
    """
    Setter for self.num_inputs
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of CloneNodes must be either 0 "
                           "or 1")
    self._num_declared_inputs = value

  @ANode.num_outputs.setter
  def num_outputs(self, value):
    """
    Setter for self.num_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of this CloneNode must be "
                           "at most", self.num_expected_outputs)
    self._num_declared_outputs = value
    
  def _update_when_linked_as_node2(self):
    """
    Fill _oslot_to_shape with shape clones of the input  
    
    NOTE: Before building a node the shapes of all directed edges should be
    specified. That is, the dictionaries _islot_to_shape and _oslot_to_shape
    must be filled for all islots and oslots of the graph nodes. If an oshape
    cannot be provided at node initialization but needs to be inferred from the
    node's inputs, then this method is called by the Builder to do so after
    declaration of the directed link
    """
    for oslot in range(self.num_expected_outputs):
      self._oslot_to_shape[oslot] = self._islot_to_shape[0]
    
  def _build(self):
    """
    Build the CloneNode
    
    Clone the input self.num_clones times and assign each clone to a different
    oslot
    """
    x_in = self._islot_to_itensor[0]
    for _ in range(self.num_clones):
      name = ( x_in.name + '_clone_' + str(self.num_outputs) if self.name is None
               else self.name )
      self._oslot_to_otensor[self.num_outputs] = tf.identity(x_in, name)
      self.num_outputs += 1
    
    self._is_built = True


if __name__ == '__main__':
  print(dist_dict)
    