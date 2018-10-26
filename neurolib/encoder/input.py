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

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode
from abc import abstractmethod

# pylint: disable=bad-indentation, no-member

data_type_dict = {'float32' : tf.float32,
                  'int32' : tf.int32}

class InputNode(ANode):
  """
  An abstract ANode representing inputs to the Model Graph (MG).
  
  An InputNode represents a source of information. InputNodes are used to
  represent user-provided data to be fed to the MG by means of a tensorflow
  Placeholder. InputNodes represent as well any random input to the MG.
  
  InputNodes have no inputs, that is, InputNodes are sources, information is
  "created" at the InputNode. Assignment to self.num_inputs is therefore
  forbidden.
  
  InputNodes have one main output and possibly secondary ones. The latter are
  used most often to output the relevant statistics of a random input. In that
  case, the main output is a sample from the corresponding distribution. The
  main output of a stochastic InputNode MUST be assigned to oslot = 0
  """
  num_expected_inputs = 0
  
  def __init__(self,
               builder,
               state_size,
               is_sequence=False):
    """
    Initialize the InputNode
    
    Args:
      label (int): A unique integer identifier for the node. 
      
      state_size (int or list of ints): The shape of the main output
          code. This excludes the 0th dimension - batch size - and the 1st
          dimension when the data is a sequence - number of steps.
      
      name (str): A unique name that identifies this node.
      
      batch_size (int): The output batch size. Set to None by default
          (unspecified batch_size)
          
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes.
    """
    super(InputNode, self).__init__()

    self.builder = builder
    self.label = builder.num_nodes
    builder.num_nodes += 1
  
    self.state_size = state_size
    self.batch_size = builder.batch_size
    self.max_steps = builder.max_steps if hasattr(builder, 'max_steps') else None
    self.is_sequence = is_sequence

    # Deal with sequences
    self.main_oshape, self.D = self.get_main_oshape(self.batch_size,
                                                    self.max_steps,
                                                    state_size)
    self._oslot_to_shape[0] = self.main_oshape
    
    if any([i is None for i in self.main_oshape]):
      self.dummy_ph = tf.placeholder(tf.float32, self.main_oshape, 'dummy')
  
  @abstractmethod
  def _build(self):
    """
    Build the InputNode.
    """
    raise NotImplementedError("Please implement me")
    

class PlaceholderInputNode(InputNode):
  """
  The PlaceholderInputNode represents data to be fed to the Model Graph,
  for example, for training or sampling purposes.
  
  PlaceholderInputNodes have a single output slot that maps to a tensorflow
  Placeholder.
 
  Class attributes:
    num_expected_outputs = 1
    num_expected_inputs = 0
  """
  num_expected_outputs = 1
  
  def __init__(self,
               builder,
               state_size,
               is_sequence=False,
               name=None,
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
    super(PlaceholderInputNode, self).__init__(builder,
                                               state_size,
                                               is_sequence=is_sequence)
    self.name = name or "In_" + str(self.label)

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
               builder,
               state_size,
               is_sequence=False,
               name=None,
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
    super(NormalInputNode, self).__init__(builder,
                                          state_size,
                                          is_sequence=is_sequence)
    self.name = "Normal_" + str(self.label) if name is None else name

    self._update_default_directives(**dirs)
    self.free_oslots = list(range(self.num_expected_outputs))
    
    self._declare_secondary_outputs()
    self.dist = None

  def _update_default_directives(self, **dirs):
    """
    Update the node directives
    
    TODO: This is only valid for 1D state_size
    """
    if self.D == 1:
#       oshape = self.main_oshape[-1:]
      oshape = self.main_oshape[1:]
    else:
      raise NotImplementedError("")
    
    if self.is_sequence and self.max_steps is None:
      mean_init = tf.zeros(oshape)
      scale = tf.eye(self.state_size)
      scale_init = tf.linalg.LinearOperatorFullMatrix(scale)
    else:
      dummy = tf.placeholder(tf.float32, oshape, 'dummy')
      mean_init = tf.zeros_like(dummy)
      scale = tf.eye(self.state_size)
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
  
  def _get_sample(self):
    """
    """
    return self.dist.sample(sample_shape=self.batch_size)
  
  def __call__(self):
    """
    """
    return self._get_sample()
  
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
    self._oslot_to_otensor[0] = dist.sample(sample_shape=self.batch_size,
                                            name=self.name)
#     assert tf.shape(self._oslot_to_otensor[0]).as_list() == self._oslot_to_shape[0] 
    assert self._oslot_to_otensor[0].shape.as_list() == self._oslot_to_shape[0] 
    
    self._oslot_to_otensor[1] = dist.loc
#     assert tf.shape(self._oslot_to_otensor[1]).as_list() == self._oslot_to_shape[1]
    assert self._oslot_to_otensor[1].shape.as_list() == self._oslot_to_shape[1]
    
    self._oslot_to_otensor[2] = dist.scale.to_dense()
#     assert tf.shape(self._oslot_to_otensor[2]).as_list() == self._oslot_to_shape[2]
    assert self._oslot_to_otensor[2].shape.as_list() == self._oslot_to_shape[2]

    self._is_built = True


if __name__ == '__main__':
  print(dist_dict.keys())  # @UndefinedVariable