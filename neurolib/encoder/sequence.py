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

import numpy as np
import tensorflow as tf

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import act_fn_dict, layers_dict, cell_dict

# pylint: disable=bad-indentation, no-member, protected-access

# class Sequence(InnerNode):
#   """
#   """
#   def __init__(self, label, num_features, max_steps, batch_size):
#     """
#     """
#     self.label = label
#     self.num_features = num_features
#     self.max_steps = max_steps
#     self.batch_size = batch_size
#     super(Sequence, self).__init__(label)
#     
#     self.main_oshape = [batch_size, max_steps, num_features]
#     self._oslot_to_shape[0] = self.main_oshape
#     
#     self.free_oslots = list(range(self.num_expected_outputs))
#     
#   @abstractmethod
#   def _build(self):
#     """
#     """
#     raise NotImplementedError("Please implement me.")
    
    
class EvolutionSequence(InnerNode):
  """
  """
  num_expected_outputs = 1
  _requires_builder = True
  def __init__(self,
               label, 
               num_features,
               init_states=None,
               num_islots=2,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward'):
    """
    Initialize a Sequence with an RNN cell
    """
    self.label = label
    self.name = 'EvSeq_' + str(label) if name is None else name    
    super(EvolutionSequence, self).__init__(label)

    self.num_features = num_features
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.main_oshape = [batch_size, max_steps, num_features]
    self._oslot_to_shape[0] = self.main_oshape
    
    if init_states is None:
      raise ValueError("`init_states` must be provided") 
    
    self.free_oslots = list(range(self.num_expected_outputs))

    self.builder = builder
    self.mode = mode
    self.num_expected_inputs = num_islots
    
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")


class BasicRNNEvolutionSequence(EvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=2,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward',
               **dirs):
    """
    """
    super(BasicRNNEvolutionSequence, self).__init__(label,
                                                    num_features,
                                                    init_states=init_states,
                                                    num_islots=num_islots,
                                                    max_steps=max_steps,
                                                    batch_size=batch_size,
                                                    name=name,
                                                    builder=builder,
                                                    mode=mode)
    if len(init_states) != 1:
      raise ValueError("`len(init_states) != 1")
    if num_features != init_states[0].num_features:
      raise ValueError("num_features != init_states.num_features, {} != {}",
                       num_features, init_states.num_features)

    if isinstance(init_states[0], str):
      self.init_state = builder.nodes[init_states]
    else:
      self.init_state = init_states[0]
    builder.addDirectedLink(self.init_state, self, islot=0)
    self._update_default_directives(**dirs)
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {'cell' : 'basic'}
    self.directives.update(dirs)
    
    self.directives['cell'] = cell_dict[self.directives['cell']]
    
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    print("sorted_inputs", sorted_inputs)
    init_state = sorted_inputs[0][1]
    inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    print("self.num_features", self.num_features)
    
    rnn_cell = self.directives['cell']
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = rnn_cell(self.num_features)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("init_state", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
      print("states_series", states_series)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True


class LSTMEvolutionSequence(EvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states=None,
               num_islots=3,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward',
               **dirs):
    super(LSTMEvolutionSequence, self).__init__(label,
                                                num_features,
                                                init_states=init_states,
                                                num_islots=num_islots,
                                                max_steps=max_steps,
                                                batch_size=batch_size,
                                                name=name,
                                                builder=builder,
                                                mode=mode)
    
    self.init_state, self.init_hidden_state = init_states[0], init_states[1]
    builder.addDirectedLink(self.init_state, self, islot=0)
    builder.addDirectedLink(self.init_hidden_state, self, islot=1)
    
    self._update_default_directives(**dirs)

  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {'cell' : 'lstm'}
    self.directives.update(dirs)
    
    self.directives['cell'] = cell_dict[self.directives['cell']]
    
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    print("sorted_inputs", sorted_inputs)
    init_state = tf.nn.rnn_cell.LSTMStateTuple(sorted_inputs[0][1], sorted_inputs[1][1])
    
    inputs_series = tuple(zip(*sorted_inputs[2:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    print("self.num_features", self.num_features)
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_features)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("init_state", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
      print("states_series", states_series)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True

# class RNN(Sequence):
#   """
#   TODO: Fix the docs
#   
#   A DeterministicNNNode (Neural Net) is a deterministic mapping with a single
#   input a single output. It is the simplest node that embodies a transformation
#   to the way information is represented.
#   
#   
#   Class attributes:
#     num_expected_inputs = 1
#     num_expected_outputs = 1    
#   """
#   _requires_builder = True
#   num_expected_outputs = 1
#   
#   def __init__(self, label, output_shape, init_state, islots,
#                batch_size=None,
#                name=None,
#                builder=None,
#                **dirs):
#     """
#     TODO: Fix the docs
# 
#     Initialize a DeterministicNNNode.
#     
#     Args:
#       label (int): A unique integer identifier for the node
#       
#       output_shape (int or list of ints): The shape of the output encoding.
#           This excludes the 0th dimension - batch size - and the 1st dimension
#           when the data is a sequence - number of steps
#           
#       name (str): A unique string identifier for this node
#     
#       batch_size (int): The output batch size. Set to None by default
#           (unspecified batch_size)
#           
#       builder (Builder): An instance of Builder necessary to declare the
#           secondary output nodes
#           
#       dirs (dict): A set of user specified directives for constructing this
#           node
#     """
#     self.name = "Det_" + str(label) if name is None else name
#     self.num_expected_inputs = islots
#     self.init_state = init_state
#     super(RNN, self).__init__(label)
#     
#     self._is_numsteps_static = True
#     self._is_sequence = False
#     if isinstance(output_shape, int):
#       main_oshape = [batch_size] + [output_shape]
#     elif isinstance(output_shape, list):
# #       if len(output_shape) == 1:
# #         self._is_sequence = False
#       if len(output_shape) == 2:
#         self._is_sequence = True
#         if output_shape[0] is None:
#           self._is_numsteps_static = False
#       if len(output_shape) > 2:
#         raise NotImplementedError
#       main_oshape = [batch_size] + output_shape
#     else:
#       raise ValueError("The output_shape of a DeterministicNNNode must be an int or "
#                        "a list of ints")
#     self.main_oshape = self._oslot_to_shape[0] = main_oshape
#     
#     self.builder = builder
#     
#     self._update_directives(**dirs)
#     
#     self._declare_init_state(init_state)
# 
#   def _update_directives(self, **dirs):
#     """
#     Update the node directives
#     """
#     pass
#   
#   def _declare_init_state(self, init_state):
#     """
#     """
#     xdim = self.main_oshape[-1]
#     idim = init_state._oslot_to_shape[0][-1]
#     if idim != xdim:
#       raise ValueError("Input dimension/code "
#                        "dimension mismatch ({}, {})".format(idim, xdim) )
#     self._islot_to_shape[0] = init_state._oslot_to_shape[0]
#     
#   def _update_when_linked_as_node2(self):
#     """
#     TODO: Fix the docs
#     
#     The input assigned to islot=1 must have the same dimension as the RNN
#     output. Check that this is indeed the case
#     
#     NOTE: Before building a node the shapes of all directed edges should be
#     specified. That is, the dictionaries _islot_to_shape and _oslot_to_shape
#     must be filled for all islots and oslots of the graph nodes. If an oshape
#     cannot be provided at node initialization but needs to be inferred from the
#     node's inputs, then this method is called by the Builder to do so after
#     declaration of the directed link
#     """
#     pass
# 
#   def _build(self):
#     """
#     """
#     z0 = self._islot_to_itensor[0][-1]
#     self._oslot_to_otensor[0].append(tf.expand_dims(z0, axis=1))
#     
#     
#     x_in = [value for value in self._islot_to_itensor.values()]
#     x_in = tf.concat(x_in, axis=2)
#     
#     cell = self.dirs['cell']
#     outputs, state = tf.nn.dynamic_rnn(cell, x_in)
#     
#     output = tf.stack(outputs, axis=1)
#     output_name = self.name + '_out' # + str(oslot) ?
#     self._oslot_to_otensor[0] = tf.identity(output, output_name) 
#       
#     self._is_built = True
# 
# 
#   def _build_old(self):
#     """
#     Build the node
#     
#     A) Scan the directives for the properties of the feedforward network
#     
#     B) Builds the tensorflow graph from the found directives
#     """
#     dirs = self.directives
#     
#     x_in = self._islot_to_itensor[0]
#     nsteps = tf.shape(x_in)[-1]
#     z0 = self._islot_to_itensor[1]
#     
#     xdim = self._islot_to_shape[-1]
#     zdim = self.main_oshape[-1]
#     # Stage A
# #     try:
# #       if 'layers' in dirs:
# #         num_layers = len(dirs['layers'])
# #         layers = [layers_dict[dirs['layers'][i]] for i in range(num_layers)]
# #       else:
# #         num_layers = dirs['num_layers']
# #         layers = [layers_dict['full'] for i in range(num_layers)]
# #       if 'activations' in dirs:
# #         activations = [act_fn_dict[dirs['activations'][i]] 
# #                                       for i in range(num_layers)]
# #       else:
# #         activation = dirs['activation']
# #         activations = [act_fn_dict[activation]
# #                                       for i in range(num_layers-1)]
# #         activations.append(None)
# #       num_nodes = dirs['num_nodes']
# #       net_grow_rate = dirs['net_grow_rate']
# #     except AttributeError as err:
# #       raise err
#   
#     # Stage B
# #     for n, layer in enumerate(layers):
# #       output_dim = self._oslot_to_shape[0][-1]
# #       if n == 0:
# #         hid_layer = layer(x_in, num_nodes, activation_fn=activations[n])
# #       elif n == num_layers-1:
# #         output = layer(hid_layer, output_dim, activation_fn=activations[n])
# #       else:
# #         hid_layer = layer(hid_layer, int(num_nodes*net_grow_rate),
# #                           activation_fn=activations[n])
# #     batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
# #     batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
#     
# #     init_state = tf.placeholder(tf.float32, [batch_size, state_size])
#     
#     W = tf.Variable(np.random.rand(xdim + zdim, zdim), dtype=tf.float32)
#     b = tf.Variable(np.zeros((1, zdim)), dtype=tf.float32)
#         
#     # Unpack columns
#     inputs_series = tf.split(x_in, nsteps, axis=1)
# 
# #     inputs_series = tf.unpack(batchX_placeholder, axis=1)
# #     labels_series = tf.unpack(batchY_placeholder, axis=1)
#     
#     # Forward pass
#     current_state = z0
#     states_series = []
#     for current_input in inputs_series:
#       current_input = tf.reshape(current_input, [self.batch_size, xdim])
#       input_and_state_concatenated = tf.concat([current_input, current_state],
#                                                axis=1)  # Increasing number of columns
#   
#       next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
#       states_series.append(next_state)
#       current_state = next_state
#     
#     output = tf.stack(states_series, axis=1)
#     output_name = self.name + '_out' # + str(oslot) ?
#     self._oslot_to_otensor[0] = tf.identity(output, output_name) 
#       
#     self._is_built = True
