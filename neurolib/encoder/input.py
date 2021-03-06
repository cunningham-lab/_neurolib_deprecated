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

# pylint: disable=bad-indentation, no-member

data_type_dict = {'float32' : tf.float32,
                  'int32' : tf.int32}

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
  
  def __init__(self,
               label,
               num_features,
               batch_size=None,
               max_steps=None,
               is_sequence=False,
               builder=None):
    """
    Initialize the InputNode
    
    Args:
      label (int): A unique integer identifier for the node. 
      
      num_features (int or list of ints): The shape of the main output
          code. This excludes the 0th dimension - batch size - and the 1st
          dimension when the data is a sequence - number of steps.
      
      name (str): A unique name that identifies this node.
      
      batch_size (int): The output batch size. Set to None by default
          (unspecified batch_size)
          
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes.
    """
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.is_sequence = is_sequence
    self.builder = builder
    self.num_features = num_features

    super(InputNode, self).__init__(label)
        
    # Deal with sequence or no sequence
    if is_sequence:
      self.main_oshape= [batch_size, max_steps, num_features]
    else:
      self.main_oshape = [batch_size, num_features]
    self._oslot_to_shape[0] = self.main_oshape
    
    if any([i is None for i in self.main_oshape]):
      self._dummy_ph = tf.placeholder(tf.float32, self.main_oshape,
                                      self.name + '_dummy')

    # Add visualization
    self.vis = pydot.Node(self.name)
  
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
  
  def __init__(self,
               label,
               num_features,
               max_steps=None,
               batch_size=None,
               builder=None,
               name=None,
               is_sequence=False,
               **dirs):
    """
    Initialize the PlaceholderInputNode
    
    Args:
      label (int): A unique integer identifier for the node.:
      main_oshape (int or list of ints): The shape of the output encoding.
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
#     self.name = "In_" + str(label) if name is None else name
    self.name = name or "In_" + str(label)
    super(PlaceholderInputNode, self).__init__(label,
                                               num_features,
                                               batch_size=batch_size,
                                               max_steps=max_steps,
                                               is_sequence=is_sequence,
                                               builder=builder)
    self._update_default_directives(**dirs)

    self.free_oslots = list(range(self.num_expected_outputs))

  def _update_default_directives(self, **dirs):
    """
    Update default directives
    """
    self.directives = {'data_type' : 'float32'}
    self.directives.update(dirs)
    
    self.directives['data_type'] = data_type_dict[self.directives['data_type']]

  def _build(self):
    """
    Build a PlaceholderInputNode.
    
    Assigns a new placeholder to _oslot_to_otensor[0]
    """
    name = self.name
    out_shape = self.main_oshape
    print("out_shape", out_shape)
    self._oslot_to_otensor[0] = tf.placeholder(self.directives['data_type'],
                                               shape=out_shape,
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

  def __init__(self,
               label,
               num_features,
               builder,
               batch_size=1,
               max_steps=None,
               name=None,
               is_sequence=False,
               **dirs):
    """
    Initialize the NormalInputNode. A builder object argument is MANDATORY in
    order to declare the secondary outputs.
    
    Args:
      label (int): A unique integer identifier for the node.
      num (int or list of ints): The shape of the output encoding.
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
    self.name = "Normal_" + str(label) if name is None else name
    super(NormalInputNode, self).__init__(label,
                                          num_features,
                                          batch_size=batch_size,
                                          max_steps=max_steps,
                                          is_sequence=is_sequence,
                                          builder=builder)
    self._update_default_directives(**dirs)

    self.free_oslots = list(range(self.num_expected_outputs))
    self._declare_secondary_outputs()
    
  def _update_default_directives(self, **dirs):
    """
    Update the node directives
    """
    oshape = self.main_oshape[1:]
    
    if self.is_sequence and self.max_steps is None:
      print("oshape, self.num_features", oshape, self.num_features)
      mean_init = tf.zeros(oshape)
      scale= tf.eye(self.num_features)
      scale_init = tf.linalg.LinearOperatorFullMatrix(scale)
    else:
#       dummy = tf.placeholder(tf.float32, [self.batch_size, self.num_features], 'd')
      dummy = tf.placeholder(tf.float32, oshape, 'd')
      mean_init = tf.zeros_like(dummy)
#       nsteps = tf.shape(mean_init)[0]
#       scale = tf.eye(self.num_features, batch_shape=[nsteps])
      scale = tf.eye(self.num_features)
      scale_init = tf.linalg.LinearOperatorFullMatrix(scale)

    self.directives = {'output_mean_name' : self.name + '_mean',
                       'output_scale_name' : self.name + '_scale',
                       'mean_init' : mean_init,
                       'scale_init' : scale_init}
    self.directives.update(dirs)
        
  def _declare_secondary_outputs(self):
    """
    Declare the statistics of the normal as secondary outputs. 
    """
    oshape = self.main_oshape[1:]
    self._oslot_to_shape[1] = oshape # mean oslot
    o1 = self.builder.addOutput(name=self.directives['output_mean_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    self._oslot_to_shape[2] = oshape[:-1] + [oshape[-1]]*2 # stddev oslot  
    o2 = self.builder.addOutput(name=self.directives['output_scale_name'])
    self.builder.addDirectedLink(self, o2, oslot=2)
    
  def _build(self):
    """
    Builds a NormalInputNode.
    
    Assigns a sample from self.dist to _oslot_to_otensor[0] 
    Assigns the mean from self.dist to _oslot_to_otensor[1]
    Assigns a cholesky decomposition of the covariance from self.dist to
    _oslot_to_otensor[2]
    """
    mean = self.directives['mean_init']
    scale = self.directives['scale_init']
    self.dist = dist = dist_dict['MultivariateNormalLinearOperator'](loc=mean,
                                                                     scale=scale)
    
    dummy = tf.placeholder(tf.float32, [self.batch_size], 'dummy')
#     out_shape = tf.shape(dummy)
    self._oslot_to_otensor[0] = dist.sample(sample_shape=self.batch_size,
                                            name=self.name)
    print("shapes compare:", tf.shape(self._oslot_to_otensor[0]),
          self._oslot_to_shape[0])
    self._oslot_to_otensor[1] = dist.loc
    print("shapes compare:", tf.shape(self._oslot_to_otensor[1]),
          self._oslot_to_shape[1])
    self._oslot_to_otensor[2] = dist.scale.to_dense()
    print("shapes compare:", tf.shape(self._oslot_to_otensor[2]),
          self._oslot_to_shape[2])

    self._is_built = True


if __name__ == '__main__':
  print(dist_dict.keys())  # @UndefinedVariable