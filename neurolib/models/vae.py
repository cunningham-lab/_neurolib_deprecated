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

from neurolib.models.models import Model

from neurolib.trainers.trainer import GDTrainer
from neurolib.builders.static_builder import StaticBuilder
from neurolib.trainers import cost_dict
from neurolib.encoder.normal import NormalTriLNode
from neurolib.utils.graphs import get_session

# pylint: disable=bad-indentation, no-member, protected-access

class VariationalAutoEncoder(Model):
  """
  The Static Variational Autoencoder.   
  """
  def __init__(self,
               latent_dim=None,
               output_dim=None,
               batch_size=1,
               builder=None,
               **dirs):
    """
    Initialize the static variational autoencoder
    """
    self.latent_dim = latent_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    
    # The main scope for this model. 
    self._main_scope = 'VariationalAutoEncoder'

    super(VariationalAutoEncoder, self).__init__()
    self.builder = builder
    if builder is not None:
      self._help_build()
    else:
      if latent_dim is None or output_dim is None:
        raise ValueError("latent_dim, output_dim are mandatory "
                         "in the default build")
        
    self._update_default_directives(**dirs)

    # Initialize at build
    self._adj_list = None
    self.bender = None
    self.cost = None
    self.nodes = None
    self.model_graph = None
      
  def _update_default_directives(self, **dirs):
    """
    Update the default specs with the ones provided by the user.
    """
    self.directives = {'num_layers_0' : 2,
                       'num_nodes_0' : 128,
                       'activation_0' : 'leaky_relu',
                       'net_grow_rate_0' : 1.0,
                       'share_params' : False,
                       'trainer' : 'gd',
                       'cost' : 'elbo',
                       'gd_optimizer' : 'adam',
                       'node_class' : NormalTriLNode}
    self.directives['cost'] = cost_dict[self.directives['cost']]  # @UndefinedVariable
    self.directives.update(dirs)
    
  def build(self):
    """
    Builds the VariationalAutoEncoder.
    
    => E =>
    """
    dirs = self.directives
    builder = self.builder
    if builder is None:
      self.builder = builder = StaticBuilder(scope=self.main_scope,
                                                  batch_size=self.batch_size)
      
      enc0 = builder.addInner(self.output_dim, name='Generative',
                              node_class=dirs['node_class'],
                              directives=dirs)
      i1 = builder.addInput(self.output_dim, name='response', **dirs)
      enc1 = builder.addInner(self.latent_dim, name='Recognition',
                              node_class=dirs['node_class'],
                              directives=dirs)
      o1 = builder.addOutput(name='copy')

      builder.addDirectedLink(i1, enc1)
      builder.addDirectedLink(enc1, enc0, oslot=0)
      builder.addDirectedLink(enc0, o1, oslot=0)      
    
      self._adj_list = builder.adj_list
    else:
      self._check_build()
      builder.scope = self.main_scope

    # Build the tensorflow graph
    self.nodes = self.builder.nodes
    builder.build()
    self.model_graph = builder.model_graph
    
    self.cost = dirs['cost'](self.nodes)  #pylint: disable=not-callable
      
    self.bender = GDTrainer(self.cost)
    
    self._is_built = True
    
  def _check_build(self):
    """
    """
    pass
  
  def train(self, dataset, num_epochs=100):
    """
    Trains the model. 
    
    The dataset provided by the client should have keys
    
    train_features, train_response
    valid_features, valid_response
    test_features, test_response
    
    where # is the number of the corresponding Input node, see model graph.
    """
    self._check_dataset_correctness(dataset)
    train_dataset, _, _ = self.make_datasets(dataset)
    batch_size = self.builder.batch_size

    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
      self.bender.update(sess,
                         tuple(zip(*train_dataset.items())),
                         batch_size=batch_size)
#       cost = np.mean(sess.run([self.cost], feed_dict=train_dataset))
      cost = self.reduce_op_from_batches(sess, [self.cost], train_dataset)
      print(cost)
      
    sess.close()
    
  def visualize_model_graph(self, filename="model_graph"):
    """
    Generates a representation of the computational graph
    """
    self.model_graph.write_png(filename)
    
  def _check_dataset_correctness(self, dataset):
    """
    """
    pass