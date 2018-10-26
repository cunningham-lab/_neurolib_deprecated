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
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected  #pylint: disable=no-name-in-module

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import MultivariateNormalTriL  # @UnresolvedImport
from neurolib.utils.utils import basic_concatenation

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu}

# pylint: disable=bad-indentation, no-member, protected-access

class NormalTriLNode(InnerNode):
  """
  A Gaussian node.
  
  
  """
  num_expected_outputs = 3
  
  def __init__(self,
               builder,
               state_size,
               num_inputs=1,
               is_sequence=False,
               name=None,
               **dirs):
    """
    Initialize a NormalInputNode
    
    Args:
      label (int): A unique identifier for the node
      state_size (int): The size of the last dimension.
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes
      name (str): A unique string identifier for this node
      batch_size (int): Self-explanatory.
      dirs (dict): A set of user specified directives for constructing this
          node
    """
    super(NormalTriLNode, self).__init__(builder,
                                         is_sequence)
    
    self.num_expected_inputs = num_inputs
    self.state_size = state_size
    self.main_oshape, self.D = self.get_main_oshape(self.batch_size,
                                            self.max_steps,
                                            state_size) 
    self._oslot_to_shape[0] = self.main_oshape
    
    self.name = "NormalTril_" + str(self.label) if name is None else name

    self._update_directives(**dirs)
    self.free_oslots = list(range(self.num_expected_outputs))
    
    self._declare_secondary_outputs()
    
    self.dist = None

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {'num_layers' : 2,
                      'num_nodes' : 128,
                      'activation' : 'leaky_relu',
                      'net_grow_rate' : 1.0,
                      'share_params' : False,
                      'output_mean_name' : self.name + '_mean',
                      'output_cholesky_name' : self.name + '_cholesky'}
    self.directives.update(dirs)
    
    # Deal with directives that map to tensorflow objects hidden from the client
    self.directives['activation'] = act_fn_dict[self.directives['activation']]
    
  def _declare_secondary_outputs(self):
    """
    Declare outputs for the statistics of the distribution (mean and standard
    deviation)
    """
    main_oshape = self._oslot_to_shape[0]
    print("main_oshape", main_oshape)
    
    # Mean oslot
    self._oslot_to_shape[1] = main_oshape
    o1 = self.builder.addOutput(name=self.directives['output_mean_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    # Stddev oslot
    self._oslot_to_shape[2] = main_oshape.append(self.state_size)  
    o2 = self.builder.addOutput(name=self.directives['output_cholesky_name'])
    
    print('_oslot_to_shape', self._oslot_to_shape)
    self.builder.addDirectedLink(self, o2, oslot=2)

  def _get_mean(self, _input):
    """
    """
    dirs = self.directives
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']
    
    output_dim = self._oslot_to_shape[0][-1]
    with tf.variable_scope(self.name+'_mean', reuse=tf.AUTO_REUSE):
      # Define the Means
      hid_layer = fully_connected(_input, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      for _ in range(num_layers-1):
        num_nodes = int(num_nodes*net_grow_rate)
        hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      mean = fully_connected(hid_layer, output_dim, activation_fn=None)
    
    return mean, hid_layer
  
  def _get_scale_tril(self, _input, hid_layer=None):
    """
    """
    dirs = self.directives
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']

    output_dim = self._oslot_to_shape[0][-1]
    with tf.variable_scope(self.name+'_scale', reuse=tf.AUTO_REUSE):
      if dirs['share_params']:
        output_chol = fully_connected(hid_layer, output_dim**2, activation_fn=None)
      else:
        print("_input:", _input)
        hid_layer = fully_connected(_input, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
        for _ in range(num_layers-1):
          num_nodes = int(num_nodes*net_grow_rate)
          hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
        output_chol = fully_connected(hid_layer, output_dim**2,
              activation_fn=None,
              weights_initializer=tf.random_normal_initializer(stddev=1e-4),
              biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(output_dim**2)))
  #           normalizer_fn=lambda x : x/tf.sqrt(x**2),
      output_chol = tf.reshape(output_chol, 
  #                              shape=[self.batch_size, output_dim, output_dim])
                               shape=[-1, output_dim, output_dim])
    return output_chol
  
  def _get_sample(self, inputs=None, islot_to_itensor=None):
    """
    """
    if inputs is not None:
      _input = basic_concatenation(inputs)
    else:
      _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_mean(_input)
    output_chol = self._get_scale_tril(_input, hid_layer)
    return MultivariateNormalTriL(loc=mean, scale_tril=output_chol).sample()
      
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    """
#     with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):    
    return self._get_sample(inputs, islot_to_itensor)
     
  def _build(self, islot_to_itensor=None):
    """
    Builds the graph corresponding to a NormalTriL encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
#     dirs = self.directives
#     if islot_to_itensor is None:
    islot_to_itensor = self._islot_to_itensor

#     num_layers = dirs['num_layers']
#     num_nodes = dirs['num_nodes']
#     activation = dirs['activation']
#     net_grow_rate = dirs['net_grow_rate']

    _input = basic_concatenation(islot_to_itensor)
#     itensors = list(zip(*sorted(islot_to_itensor.items())))[1] # make sure the inputs are ordered
#     _input = tf.concat(itensors, axis=-1)
#       _input = self._islot_to_itensor[0]
#     output_dim = self._oslot_to_shape[0][-1] # Last dim

#     with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      # Define the Means
    mean, hid_layer = self._get_mean(_input)
#       hid_layer = fully_connected(_input, num_nodes, activation_fn=activation,
#             biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
#       for _ in range(num_layers-1):
#         num_nodes = int(num_nodes*net_grow_rate)
#         hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
#             biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
#       mean = fully_connected(hid_layer, output_dim, activation_fn=None)
    
      # Define the Cholesky Lower Decomposition
#       if dirs['share_params']:
#         output_chol = fully_connected(hid_layer, output_dim**2, activation_fn=None)
#       else:
#         hid_layer = fully_connected(_input, num_nodes, activation_fn=activation,
#             biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
#         for _ in range(num_layers-1):
#           num_nodes = int(num_nodes*net_grow_rate)
#           hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
#             biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
#         output_chol = fully_connected(hid_layer, output_dim**2,
#             activation_fn=None,
#             weights_initializer = tf.random_normal_initializer(stddev=1e-4),
#   #           normalizer_fn=lambda x : x/tf.sqrt(x**2),
#             biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(output_dim**2)))
#       output_chol = tf.reshape(output_chol, 
#   #                              shape=[self.batch_size, output_dim, output_dim])
#                                shape=[-1, output_dim, output_dim])
      
    output_chol = self._get_scale_tril(_input, hid_layer)
    if 'output_mean_name' in self.directives:
      mean_name = self.directives['output_mean_name']
    else:
      mean_name = "Mean_" + str(self.label) + '_0'
    if 'output_cholesky_name' in self.directives:
      cholesky_name = self.directives['output_cholesky_name']
    else:
      cholesky_name = 'CholTril_' + str(self.label) + '_0'
    
    cholesky_tril = tf.identity(output_chol, name=cholesky_name)
    
    # Get the tensorflow distribution for this node
    self.dist = MultivariateNormalTriL(loc=mean, scale_tril=cholesky_tril)

    # Fill the oslots
    self._oslot_to_otensor[0] = self.dist.sample(name='Out' + 
                                                 str(self.label) + '_0')
    self._oslot_to_otensor[1] = tf.identity(mean, name=mean_name)
    self._oslot_to_otensor[2] = cholesky_tril
    
    self._is_built = True
    
  def _log_prob(self, ipt):
    """
    Define the loglikelihood of the distribution
    """    
    return self.dist.log_prob(ipt)