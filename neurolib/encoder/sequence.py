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
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell

# pylint: disable=bad-indentation, no-member, protected-access

class EvolutionSequence(InnerNode):
  """
  An EvolutionSequence represents a sequence of mappings, each with the
  distinguishining feature that it takes the output of their predecessor as
  input. This makes them appropriate in particular to represent the time
  evolution of a code.
  
  RNNs are children of EvolutionSequence.
  """
  num_expected_outputs = 1
  def __init__(self,
               builder,
               state_size,
#                init_states=None,
               num_inputs=2,
               name=None,
               mode='forward'):
    """
    Initialize an EvolutionSequence
    """
    super(EvolutionSequence, self).__init__(builder,
                                            is_sequence=True)
    self.name = 'EvSeq_' + str(self.label) if name is None else name    

    self.state_size = state_size
    self.main_oshape = [self.batch_size, self.max_steps, state_size]
    self._oslot_to_shape[0] = self.main_oshape
    
#     if init_states is None:
#       raise ValueError("`init_states` must be provided") 
    
    self.free_oslots = list(range(self.num_expected_outputs))

    self.builder = builder
    self.mode = mode
    self.num_expected_inputs = num_inputs
    
  @abstractmethod
  def _build(self, islot_to_itensor):
    """
    """
    raise NotImplementedError("Please implement me.")


class BasicRNNEvolutionSequence(EvolutionSequence):
  """
  BasicRNNEvolutionSequence is the simplest possible EvolutionSequence. It is an
  evolution sequence characterized by a single latent state. In particular this
  implies that a single initial state is passed.
  
  hose inputs are an external input tensor and the previous
  state of the sequence, and whose output is the (i+1)th state.
  """
  def __init__(self,
               builder,
               state_size,
#                init_states,
               num_inputs=2,
               name=None,
               cell_class='basic',
               mode='forward',
               **dirs):
    """
    Initialize the BasicRNNEvolutionSequence
    """
    super(BasicRNNEvolutionSequence, self).__init__(builder,
                                                    state_size,
#                                                     init_states=init_states,
                                                    num_inputs=num_inputs,
                                                    name=name,
                                                    mode=mode)
#     if len(init_states) != 1:
#       raise ValueError("`len(init_states) != 1`")
#     if state_size != init_states[0].state_size:
#       raise ValueError("state_size != init_states.state_size, {} != {}",
#                        state_size, init_states.state_size)
# 
#     if isinstance(init_states[0], str):
#       self.init_inode = builder.nodes[init_states]
#     else:
#       self.init_inode = init_states[0]
    
    # Get the cell_class
    self.cell_class = cell_class = (cell_dict[cell_class] if isinstance(cell_class, str) 
                                    else cell_class) 
    
    self._update_default_directives(**dirs)

    # Add the init_inode and the init_inode -> ev_seq edge
    if issubclass(cell_class, CustomCell): 
      self.cell = cell_class(self.state_size, builder=self.builder)  #pylint: disable=not-callable
    else:
      self.cell = cell_class(self.state_size)
    self._declare_init_state()

    
  def _declare_init_state(self):
    """
    Declare the initial state of the Evolution Sequence.
    
    Uses that the BasicRNNEvolutionSequence has only one evolution input
    """
    builder = self.builder
    
    try:
      self.init_inode = self.cell.get_init_states(ext_builder=builder)[0]
      self.init_inode = builder.nodes[self.init_inode]
      builder.addDirectedLink(self.init_inode, self, islot=0)
    except AttributeError:
      self.init_inode = builder.addInput(self.state_size, iclass=NormalInputNode)
      builder.addDirectedLink(self.init_inode, self, islot=0)
      self.init_inode = builder.nodes[self.init_inode]
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
        
  def _build(self, islot_to_itensor=None):
    """
    Build the Evolution Sequence
    """
    cell = self.cell
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      init_state = self.init_inode()

      sorted_inputs = sorted(self._islot_to_itensor.items())
  #     init_inode = sorted_inputs[0][1]
      print(sorted_inputs)
      inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
      if len(inputs_series) == 1:
        inputs_series = inputs_series[0]
      else:
        inputs_series = tf.concat(inputs_series, axis=-1)
        
      print("inputs_series", inputs_series)
      print("init_state", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True


class LSTMEvolutionSequence(EvolutionSequence):
  """
  """
  def __init__(self,
               builder,
               state_size,
               init_states,
               num_inputs=3,
               name=None,
               mode='forward',
               **dirs):
    """
    Initialize the LSTMEvolutionSequence
    """
    super(LSTMEvolutionSequence, self).__init__(builder,
                                                state_size,
                                                init_states=init_states,
                                                num_inputs=num_inputs,
                                                name=name,
                                                mode=mode)
    
    self.init_inode, self.init_hidden_state = init_states[0], init_states[1]
    builder.addDirectedLink(self.init_inode, self, islot=0)
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
    
    init_state = tf.nn.rnn_cell.LSTMStateTuple(sorted_inputs[0][1], sorted_inputs[1][1])
    
    inputs_series = tuple(zip(*sorted_inputs[2:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size,
                                          state_is_tuple=True)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("sorted_inputs", sorted_inputs)
      print("state tuple", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True


class LinearNoisyDynamicsEvSeq(EvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=1,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward',
               **dirs):
    """
    """
    self.init_inode = init_states[0]
    super(LinearNoisyDynamicsEvSeq, self).__init__(label,
                                                   num_features,
                                                   init_states=init_states,
                                                   num_inputs=num_islots,
                                                   max_steps=max_steps,
                                                   batch_size=batch_size,
                                                   name=name,
                                                   builder=builder,
                                                   mode=mode)

    builder.addDirectedLink(self.init_inode, self, islot=0)
    self._update_default_directives(**dirs)
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
        
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
    print("self.state_size", self.state_size)
    
    rnn_cell = self.directives['cell']
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = rnn_cell(self.state_size)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("init_inode", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
      print("states_series", states_series)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True
    
    
class CustomEvolutionSequence():
  """
  """
  pass