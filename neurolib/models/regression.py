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

from neurolib.trainers.trainer import GDTrainer
from neurolib.builders.static_builder import StaticBuilder
from neurolib.trainers import cost_dict
from neurolib.utils.graphs import get_session

# pylint: disable=bad-indentation, no-member, protected-access

class Regression(Model):
  """
  The Regression Model implements regression with arbitrary features. It is
  specified by defining a single Model Graph (MG), with a single InputNode and a
  single OutputNode. The MG itself is an acyclic directed graph formed of any
  combination of deterministic encoders nodes.
  
  Ex: A chain of encoders with a single input and output is a Regression model:
  
  I1[ -> d_0] => E1[d_0 -> d_1] => ... => O1[d_{n} -> ]
  
  since it has a single Input node and a single Output node. The following
  directed graph, with the input flowing towards the output through 2 different
  encoder routes (rhombic) is also a Regression model:
  
  I1[ -> d_0] => E1[d_0 -> d_1], E2[d_0 -> d_2]
  
  E1[d_0 -> d_1], E2[d_0 -> d_2] => O1[d_1 + d_2 -> ]
  
  Any user defined Regression must respect the names of the mandatory Input and
  Output nodes, which are fixed to "features" and "response" respectively. 
  
  The default Regression instance builds a Model graph with just one inner
  Encoder
  
  I1[ -> d_0] => E1[d_0 -> d_1] => O1[d_{1} -> ]
  
  The inner encoder node is parameterized by a neural network which can be
  controlled through the directives. Specifically, linear regression is achieved
  by initializing Regression with num_layers=1 and activation=None
  """
  def __init__(self,
               input_dim=None,
               output_dim=1,
               builder=None,
               batch_size=1,
               **dirs):
    """
    Initialize the Regression Model
    
    Args:
      input_dim (int): The number of features (dimension of the input variable)
      output_dim (int): The output dimension
      builder (StaticBuilder): An instance of Builder used to build a custom
          Regression model
    """
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    
    self._main_scope = 'Regression'

    super(Regression, self).__init__()

    self.builder = builder
    if self.builder is None:
      if input_dim is None:
        raise ValueError("Argument input_dim is required to build the default "
                         "Regression")
      elif output_dim > 1:
        raise NotImplementedError("Multivariate regression is not implemented")
      
    self._update_default_directives(**dirs)

    # Defined on build
    self._adj_list = None
    self.nodes = None
    self._model_graph = None
    
    self.cost = None
    self.trainer = None

  def _update_default_directives(self, **directives):
    """
    Update the default directives with user-provided ones.
    """
    self.directives = {'trainer' : 'gd',
                       'loss_func' : 'mse',
                       'gd_optimizer' : 'adam',
                       }
    if self.builder is None:
      self.directives.update({'num_layers' : 2,
                              'num_nodes' : 128,
                              'activation' : 'leaky_relu',
                              'net_grow_rate' : 1.0,
                              'share_params' : False})
    
    self.directives.update(directives)
    self.directives['loss_func'] = cost_dict[self.directives['loss_func']]  # @UndefinedVariable

  def build(self):
    """
    Builds the Regression.
    
    => E =>
    """
    builder = self.builder
    dirs = self.directives
    if builder is None:
      self.builder = builder = StaticBuilder(scope=self.main_scope,
                                             batch_size=self.batch_size)
      
      in0 = builder.addInput(self.input_dim, name="features", **dirs)
      enc1 = builder.addInner(1, self.output_dim, **dirs)
      out0 = builder.addOutput(name="prediction")

      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)

      self._adj_list = builder.adj_list
    else:
      self._check_custom_build()
      builder.scope = self.main_scope
    in1 = builder.addInput(self.output_dim, name="i_response")
    out1 = builder.addOutput(name="response")
    builder.addDirectedLink(in1, out1)

    # Build the tensorflow graph
    builder.build()
    self.nodes = self.builder.nodes
    self._model_graph = builder.model_graph
    
    self.cost = dirs['loss_func'](self.nodes)  #pylint: disable=not-callable
    self.trainer = GDTrainer(self.cost, **dirs)
      
    self._is_built = True
    
  def _check_custom_build(self):
    """
    Check that the user-declared build is consistent with the Regression class
    """
    pass
    
  def _check_dataset_correctness(self, dataset):
    """
    """
    pass
  
  def train(self, dataset, num_epochs=100):
    """
    Train the Regression model. 
    
    The dataset, provided by the client, should have keys
    
    train_features, train_response
    valid_features, valid_response
    test_features, test_response
    """
    self._check_dataset_correctness(dataset)
    train_dataset, _, _ = self.make_datasets(dataset)
    batch_size = self.batch_size

    print('train_dataset.keys():', train_dataset.keys())
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
      self.trainer.update(sess,
                          tuple(zip(*train_dataset.items())),
                          batch_size=batch_size)
      cost = self.reduce_op_from_batches(sess, [self.cost], train_dataset)
      print(cost)
        
  def visualize_model_graph(self, filename="_model_graph"):
    """
    Generate a representation of the computational graph
    """
    self._model_graph.write_png(filename)
    
  def sample(self, input_data, node='prediction', islot=0):
    """
    Sample from Regression
    """
    return Model.sample(self, input_data, node, islot=islot)