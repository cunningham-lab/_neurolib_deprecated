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
import tensorflow as tf

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import act_fn_dict, layers_dict


class DeterministicNNNode(InnerNode):
  """
  A DeterministicNNNode (Neural Net) is a deterministic mapping with a single
  input a single output. It is the simplest node that embodies a transformation
  to the way information is represented.
  
  
  Class attributes:
    num_expected_inputs = 1
    num_expected_outputs = 1    
  """
  _requires_builder = False
  num_expected_inputs = 1
  num_expected_outputs = 1
  
  def __init__(self, label, output_shape, batch_size=None,
               name=None, **dirs):
    """
    Initialize a DeterministicNNNode.
    
    Args:
      label (int): A unique integer identifier for the node
      
      output_shape (int or list of ints): The shape of the output encoding.
          This excludes the 0th dimension - batch size - and the 1st dimension
          when the data is a sequence - number of steps
          
      name (str): A unique string identifier for this node
    
      batch_size (int): The output batch size. Set to None by default
          (unspecified batch_size)
          
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes
          
      dirs (dict): A set of user specified directives for constructing this
          node
    """
    self.name = "Det_" + str(label) if name is None else name
    super(DeterministicNNNode, self).__init__(label)
    
    if isinstance(output_shape, int):
      output_shape = [batch_size] + [output_shape]
    elif isinstance(output_shape, list):
      if isinstance(output_shape[0], int):
        output_shape = [batch_size] + output_shape
      elif isinstance(output_shape[0], list):
        output_shape = [batch_size] + output_shape[0]
    else:
      raise ValueError("The output_shape of a DeterministicNNNode must be an int or "
                       "a list of ints")
    self._oslot_to_shape[0] = output_shape
    
    self._update_directives(**dirs)
    
  def _update_directives(self, directives):
    """
    Update the node directives
    """
    self.directives = {'num_layers' : 2,
                       'num_nodes' : 128,
                       'activation' : 'relu',
                       'net_grow_rate' : 1.0}
    self.directives.update(directives)
    
  @InnerNode.num_inputs.setter
  def num_inputs(self, value):
    """
    Setter for self.num_inputs
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of DeterministicNNNode "
                           "should not be greather than ",
                           self.num_expected_inputs)
    self._num_declared_inputs = value

  @InnerNode.num_outputs.setter
  def num_outputs(self, value):
    """
    Setter for self.num_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of DeterministicNNNode must "
                           "should not be greather than ", self.num_expected_outputs)
    self._num_declared_outputs = value
        
  def _build(self):
    """
    Build the node
    
    A) Scan the directives for the properties of the feedforward network
    
    B) Builds the tensorflow graph from the found directives
    """
    dirs = self.directives
    
    x_in = self._islot_to_itensor[0]

    # Stage A
    try:
      if 'layers' in dirs:
        num_layers = len(dirs['layers'])
        layers = [layers_dict[dirs['layers'][i]] for i in range(num_layers)]
      else:
        num_layers = dirs['num_layers']
        layers = [layers_dict['full'] for i in range(num_layers)]
      if 'activations' in dirs:
        activations = [act_fn_dict[dirs['activations'][i]] 
                                      for i in range(num_layers)]
      else:
        activation = dirs['activation']
        activations = [act_fn_dict[activation]
                                      for i in range(num_layers-1)]
        activations.append(None)
      num_nodes = dirs['num_nodes']
      net_grow_rate = dirs['net_grow_rate']
    except AttributeError as err:
      raise err
  
    # Stage B
    for n, layer in enumerate(layers):
      output_dim = self._oslot_to_shape[0][-1]
      if n == 0:
        hid_layer = layer(x_in, num_nodes, activation_fn=activations[n])
      elif n == num_layers-1:
        output = layer(hid_layer, output_dim, activation_fn=activations[n])
      else:
        hid_layer = layer(hid_layer, int(num_nodes*net_grow_rate),
                          activation_fn=activations[n])
    
    output_name = self.name + '_out' # + str(oslot) ?
    self._oslot_to_otensor[0] = tf.identity(output, output_name) 
      
    self._is_built = True


class DeterministicCustomNode(InnerNode):
  """
  """
  pass


class Deterministic2DNNNode(InnerNode):
  """
  """
  pass