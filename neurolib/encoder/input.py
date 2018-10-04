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

import numpy as np
import tensorflow as tf

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode
from abc import abstractmethod

class InputNode(ANode):
  """
  An InputNode represents a source of information in the Model graph (MG).
  InputNodes are used to represent user-provided data to be fed to the MG, by
  means of a tensorflow Placeholder. InputNodes represent as well any random
  input to the MG.
  
  InputNodes have no inputs, that is, information is "created" at the InputNode.
  Assignment to self.num_inputs is therefore forbidden.
  
  InputNodes have one main output and possibly secondary ones. The latter are
  used most often to output the statistics of an InputNode representing a random
  input. In that case, the main output is a sample from the corresponding
  distribution. The main output MUST be assigned to oslot = 0
  """
  num_expected_inputs = 0
  
  def __init__(self, label, main_output_shape, batch_size=None,
               builder=None):
    """
    Initialize the InputNode
    
    Args:
      label (int): A unique integer identifier for the node. 
      
      main_output_shape (int or list of ints): The shape of the main output
          code. This excludes the 0th dimension - batch size - and the 1st
          dimension when the data is a sequence - number of steps.
      
      name (str): A unique name that identifies this node.
      
      batch_size (int): The output batch size. Set to None by default
          (unspecified batch_size)
          
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes.
    """
    self.batch_size = batch_size
    self.builder = builder
        
    # Deal with several possible types of output_shapes
    if isinstance(main_output_shape, int):
      main_output_shape = [batch_size] + [main_output_shape]
    elif isinstance(main_output_shape, list):
      main_output_shape = [batch_size] + main_output_shape
    else:
      raise ValueError("The main_output_shape of an InputNode must be an int or "
                       "a list of ints")
    self.main_oshape = main_output_shape
    super(InputNode, self).__init__(label)

    self._oslot_to_shape[0] = main_output_shape

    # Add visualization
    self.vis = pydot.Node(self.name)
  
  @ANode.num_inputs.setter
  def num_inputs(self, value):
    """
    Setter for num_inputs
    """
    raise AttributeError("Assignment to attribute num_inputs of InputNode is "
                         " disallowed. num_inputs is fixed to 0 for an InputNode")

  @ANode.num_outputs.setter
  def num_outputs(self, value):
    """
    Setter for num_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of InputNodes must be either 0 "
                           "or 1")
    self._num_declared_outputs = value
  
  @abstractmethod
  def _build(self):
    """
    Build the InputNode
    """
    raise NotImplementedError("Please implement me")
    

class PlaceholderInputNode(InputNode):
  """
  The PlaceholderInputNode represents data that to be fed to the Model Graph,
  usually, for training or sampling purposes. 
  
  PlaceholderInputNodes have a single output slot that maps to a tensorflow
  Placeholder.
 
  Class attributes:
    num_expected_outputs = 1
    num_expected_inputs = 0
  """
  num_expected_outputs = 1
  
  def __init__(self, label, output_shape, name=None, batch_size=None,
               **dirs):
    """
    Initialize the PlaceholderInputNode
    
    Args:
      label (int): A unique integer identifier for the node.:
      
      output_shape (int or list of ints): The shape of the output encoding.
          This excludes the 0th dimension - batch size - and the 1st dimension
          when the data is a sequence - number of steps.
          
      name (str): A unique name for this node.
      
      batch_size (int): The output batch size. Set to None by default
          (unspecified batch_size)
          
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes. 
          
      dirs (dict): A set of user specified directives for constructing this
          node.
    """
    self.name = "In_" + str(label) if name is None else name
    super(PlaceholderInputNode, self).__init__(label, output_shape,
                                               batch_size=batch_size,
                                               builder=None)
    
    self._update_directives(**dirs)

  def _update_directives(self, **dirs):
    """
    Update default directives
    """
    self.directives = {}
    self.directives.update(dirs)
    
  def _build(self):
    """
    Build a PlaceholderInputNode.
    
    Assigns a new placeholder to _oslot_to_otensor[0]
    """
    name = self.name
    out_shape = self.main_oshape
    self._oslot_to_otensor[0] = tf.placeholder(tf.float32, shape=out_shape,
                                               name=name)

    self._is_built = True


class NormalInputNode(InputNode):
  """
  A NormalInputNode represents a random input to the Model graph (MG) drawn from
  a normal distribution. 
  
  The main output (oslot=0) of a NormalInputNode is a sample. The statistics
  (possibly trainable) of the distribution are the secondary outputs.
  
  Class attributes:
    num_expected_outputs = 3
    num_expected_inputs = 0
  """  
  num_expected_outputs = 3

  def __init__(self, label, output_shape, builder, name=None, batch_size=1,
               **dirs):
    """
    Initialize the NormalInputNode. A builder object argument is MANDATORY in
    order to declare the secondary outputs.
    
    Args:
      label (int): A unique integer identifier for the node.
      
      output_shape (int or list of ints): The shape of the output encoding.
          This excludes the 0th dimension - batch size - and the 1st dimension
          when the data is a sequence - number of steps.
          
      name (str): A unique name for this node.
      
      batch_size (int): The output batch size. Set to None by default
          (unspecified batch_size)
          
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes. 
          
      dirs (dict): A set of user specified directives for constructing this
          node.
    """
    if builder is None:
      raise ValueError("A builder must be provided")
    
    self.name = "Normal_" + str(label) if name is None else name
    super(PlaceholderInputNode, self).__init__(label, output_shape,
                                               batch_size=batch_size,
                                               builder=builder)
    
    self._update_directives(**dirs)
    
    self._declare_secondary_outputs()
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.main_oshape = self.main_oshape[1:]
    
    mean_init = np.zeros(self.main_oshape)
    scale_init = np.eye(self.main_oshape[-1])
    self.directives = {'output_mean_name' : self.name + '_mean',
                       'output_cholesky_name' : self.name + '_cholesky',
                       'mean_init' : mean_init,
                       'scale_init' : scale_init}
    self.directives.update(dirs)
        
  def _declare_secondary_outputs(self):
    """
    Declare the statistics of the normal as secondary outputs. 
    """
    oshape = self.main_oshape
    self._oslot_to_shape[1] = oshape # mean oslot
    o1 = self.builder.addOutput(name=self.directives['output_mean_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    self._oslot_to_shape[2] = oshape*2 # std oslot  
    o2 = self.builder.addOutput(name=self.directives['output_cholesky_name'])
    self.builder.addDirectedLink(self, o2, oslot=2)
    
  def _build(self):
    """
    Builds a NormalInputNode.
    
    Assigns a sample from self.dist to _oslot_to_otensor[0] 
    
    Assigns the mean from self.dist to _oslot_to_otensor[1]
    
    Assigns a cholesky decomposition of the covariance from self.dist to
    _oslot_to_otensor[2]
    """
    out_shape = self.main_oshape
    
    mean = self.directives['mean_init']
    scale = self.directives['scale_init']
    self.dist = dist = dist_dict['MultivariateNormalLinearOperator'](loc=mean,
                                                                     scale=scale)
    
    self._oslot_to_otensor[0] = dist.sample(sample_shape=out_shape,
                                            name=self.name)
    self._oslot_to_otensor[1] = dist.mean()
    self._oslot_to_otensor[2] = dist.scale

    self._is_built = True


if __name__ == '__main__':
  print(dist_dict.keys())  # @UndefinedVariable
  
  