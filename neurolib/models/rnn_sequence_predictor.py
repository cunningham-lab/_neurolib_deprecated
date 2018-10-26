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

from neurolib.models.models import Model
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.encoder.input import NormalInputNode
from neurolib.trainers.trainer import GDTrainer
from neurolib.utils.graphs import get_session
from neurolib.trainers import cost_dict
from neurolib.encoder.cells import get_cell_init_states,\
  are_seq_and_cell_compatible

# pylint: disable=bad-indentation, no-member, protected-access

class RNNClassifier(Model):
  """
  """
  def __init__(self,
               num_labels,
               num_inputs=1,
               input_dim=None,
               latent_dim=None,
               builder=None,
               batch_size=1,
               max_steps=25,
               seq_class='basic',
               cell_class='basic',
               **dirs):
    """
    """
    self.input_dim = input_dim
    self.num_labels = num_labels
    self.num_inputs = num_inputs
    self.latent_dim = latent_dim
    
    self.batch_size = batch_size
    self.max_steps = max_steps
    self.cell_class = cell_class
    self.seq_class = seq_class
    are_seq_and_cell_compatible(seq_class, cell_class)
    
    self._main_scope = 'RNNClassifier'
    
    super(RNNClassifier, self).__init__()

    self.builder = builder
    if self.builder is None:
      if input_dim is None:
        raise ValueError("Argument input_dim is required to build the default "
                         "RNNClassifier")
      if latent_dim is None:
        raise ValueError("Argument latent_dim is required to build the default "
                         "RNNClassifier")
        
    self._update_default_directives(**dirs)

    # Defined on build
    self._adj_list = None
    self.nodes = None
    self._model_graph = None
    
    self.cost = None
    self.trainer = None

  def _update_default_directives(self,
                                 **directives):
    """
    Update the default directives with user-provided ones.
    """
    self.directives = {'trainer' : 'gd',
                       'loss_func' : 'cross_entropy',
                       'gd_optimizer' : 'adam',
                       'lr' : 1e-3,
                       'hidden_dims' : []}
    if self.cell_class == 'lstm':
      self.directives['hidden_dims'] = [self.latent_dim] 
    
    self.directives.update(directives)
    
    self.directives['loss_func'] = cost_dict[self.directives['loss_func']]
    self.directives['hidden_dims'].insert(0, self.latent_dim)
        
  def build(self):
    """
    Builds the RNNClassifier
    """
    builder = self.builder
    dirs = self.directives
    if builder is None:
      self.builder = builder = SequentialBuilder(scope=self.main_scope,
                                                 max_steps=self.max_steps,
                                                 batch_size=self.batch_size)
#       init_states = get_cell_init_states(builder,
#                                          self.cell_class,
#                                          dirs['hidden_dims'])
#       num_init_states = len(init_states)
#       evseq_num_inputs = self.num_inputs + num_init_states
#       iseq_islot = num_init_states

#       else:
#         cell = self.cell_class(state_size=self.latent_dim,
#                                builder=builder,
#                                **dirs)
#         init_states = cell.get_istates()
#         evseq_num_inputs = cell.num_expected_inputs
#         iseq_islot = evseq_num_inputs - 1
      is1 = builder.addInputSequence(self.input_dim, name='iseq')
      evs1 = builder.addEvolutionSequence(num_features=self.latent_dim,
#                                           num_inputs=evseq_num_inputs,
                                          num_inputs=2,
#                                           init_states=init_states,
                                          ev_seq_class=self.seq_class,
                                          cell_class=self.cell_class,
                                          name='ev_seq',
                                          **dirs)
      inn1 = builder.addInnerSequence(self.num_labels)
      os1 = builder.addOutputSequence(name='prediction')
            
#       builder.addDirectedLink(is1, evs1, islot=iseq_islot)
      builder.addDirectedLink(is1, evs1, islot=1)
      builder.addDirectedLink(evs1, inn1)
      builder.addDirectedLink(inn1, os1)
      
      self._adj_list = builder.adj_list
    else:
      self._check_custom_build()
      builder.scope = self.main_scope
    is2 = builder.addInputSequence(1, name='i_labels', data_type='int32')
    os2 = builder.addOutputSequence(name='labels')
    builder.addDirectedLink(is2, os2)
    
    builder.build()
    
    self.nodes = self.builder.nodes
    self.cost = dirs['loss_func'](self.nodes)  #pylint: disable=not-callable
    self.trainer = GDTrainer(self.cost, **dirs)
      
    self._is_built = True

  def _check_custom_build(self):
    """
    TODO:
    """
    pass
  
  def _check_dataset_correctness(self, dataset):
    """
    TODO:
    """
    pass
  
  def train(self, dataset, num_epochs=100, **dirs):
    """
    Train the RNNClassifier model. 
    
    The dataset, provided by the client, should have keys
    
    train_features, train_labels
    valid_features, valid_response
    test_features, test_response
    """
    self._check_dataset_correctness(dataset)
    train_dataset, _, _ = self.make_datasets(dataset)
    batch_size = self.batch_size
    
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
      self.trainer.update(sess,
                          tuple(zip(*train_dataset.items())),
                          batch_size=batch_size)
      cost = self.reduce_op_from_batches(sess, [self.cost], train_dataset)
      print(cost)
  
  def sample(self, input_data, node='prediction', islot=0):
    """
    """
    return Model.sample(self, input_data, node, islot=islot)