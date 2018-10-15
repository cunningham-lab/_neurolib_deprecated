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
from neurolib.utils.graphs import get_session

# pylint: disable=bad-indentation, no-member, protected-access

class Regression(Model):
  """
  The Regression Model is the simplest possible model in the Encoder paradigm.
  It consists of a single chain of encoders with a single input and output.
  
  in => E_1 => E_2 => ... => E_n => out
  
  The 
  
  Linear Regression is achieved by initializing Regression with num_layers=1 and
  activation=None
  """
  def __init__(self,
               input_dim=None,
               output_dim=None,
               builder=None,
               **dirs):
    """
    Initialize the Regression Model
    """
    self.input_dim = input_dim
    self.output_dim = output_dim
    
    # The main scope for this model. 
    self._main_scope = 'Regression'

    super(Regression, self).__init__()
    self.builder = builder
    self._update_default_directives(**dirs)
    if builder is not None:
      self._help_build()
    else:
      if input_dim is None or output_dim is None:
        raise ValueError("Both the input dimension (in_dims) and the output "
                         "dimension (out_dims) are necessary in order to "
                         "specify build the default Regression.")
      
    # Defined on build
    self._adj_list = None
    self._nodes = None
    self._model_graph = None
    
    self.cost = None
    self.bender = None


  def _update_default_directives(self, **directives):
    """
    Update the default directives with user-provided ones.
    """
    self.directives = {'trainer' : 'gd',
                       'loss_func' : 'mse',
                       'gd_optimizer' : 'adam'}
    if self.builder is None:
      self.directives.update({'num_layers' : 2,
                              'num_nodes' : 128,
                              'activation' : 'leaky_relu',
                              'net_grow_rate' : 1.0,
                              'share_params' : False})
    self.directives['loss_func'] = cost_dict[self.directives['loss_func']]  # @UndefinedVariable
    
    self.directives.update(directives)
                          
  def _help_build(self):
    """
    Check that the client-provided builder corresponds indeed to a
    Regression. 
    
    !! Not clear whether this is actually needed, keep for now.
    """
    dirs = self.directives
    trainer = dirs['trainer']
    print("Hi! I see you are attempting to build a Regressor by yourself."
          "In order for your model to be consistent with the ", trainer,
          " Trainer, you must implement the following Output Nodes:")
    if trainer == 'gd-mse':
      print("OutputNode(input_dim={}, name='regressors')".format(self.output_dim))
      print("OutputNode(input_dim={}, name='response')".format(self.output_dim))
      print("\nThis is an absolute minimum requirement and NOT a guarantee that a custom "
            "model will be successfully trained (read the docs for more).")
#     
#   def _get_directives(self):
#     """
#     Return two directives to build the two encoders that make up the default
#     Model.
#     
#     Directives:
#       nnodes_1stlayer_encj : The number of _nodes in the first hidden layer of
#             the encoder j
#       ntwrk_grow_rate_encj : The ratio between the number of _nodes of subsequent
#             layers
#       TODO: ...
#     """
#     enc_directives = self.directives
#     in_directives = {}
#     return enc_directives, in_directives

  def build(self):
    """
    Builds the Regression.
    
    => E =>
    """
    builder = self.builder
    dirs = self.directives
    if builder is None:
      self.builder = builder = StaticBuilder(scope=self.main_scope)
      
#       enc_dirs, in_dirs = self._get_directives()

      in0 = builder.addInput(self.input_dim, name="features", **dirs)
      enc1 = builder.addInner(1, self.output_dim, **dirs)
      out0 = builder.addOutput(name="prediction")

      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)

      in1 = builder.addInput(self.output_dim, name="input_response")
      out1 = builder.addOutput(name="response")
      builder.addDirectedLink(in1, out1)
    
      self._adj_list = builder.adj_list
    else:
      self._check_user_build()
      builder.scope = self.main_scope

    # Build the tensorflow graph
    builder.build()
    self._nodes = self.builder.nodes
    self._model_graph = builder.model_graph
    
#     self.cost = self._define_cost()
    self.cost = dirs['loss_func'](self._nodes)
    self.bender = GDTrainer(self.cost, **dirs)
      
    self._is_built = True
    
  def _check_user_build(self):
    """
    Check that the user-declared build is consistent with the Regression class
    """
    pass

#   def _define_cost(self):
#     """
#     TODO: Is this function necessary?
#     """
#     cost = self.directives['cost']
#     return cost(self._nodes)
    
  def update(self, dataset):
    """
    Carry a single update on the model  
    """
    self.bender.update(dataset)
  
  def _check_dataset_correctness(self, dataset):
    """
    """
    pass
  
  def train(self, dataset, num_epochs=100, batch_size=1):
    """
    Train the Regression model. 
    
    The dataset, provided by the client, should have keys
    
    train_features, train_response
    valid_features, valid_response
    test_features, test_response
    """
    self._check_dataset_correctness(dataset)
    train_dataset, _, _ = self.make_datasets(dataset)

    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
      self.bender.update(sess,
                         tuple(zip(*train_dataset.items())),
                         batch_size=batch_size)
      cost = np.mean(sess.run([self.cost], feed_dict=train_dataset))
      print(cost)
    
  def visualize_model_graph(self, filename="_model_graph"):
    """
    Generate a representation of the computational graph
    """
    self._model_graph.write_png(filename)
    
  def sample(self):
    """
    Sample from the model graph. For user provided features generates a response.
    """
    pass