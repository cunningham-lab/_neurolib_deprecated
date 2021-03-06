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

# pylint: disable=bad-indentation, no-member, protected-access

def mse(node_dict):
  """
  """
  try:
    nodeY = node_dict['response']
    nodeX = node_dict['prediction']
  except AttributeError:
    raise AttributeError("You must define two OutputNodes, named 'prediction' and "
                         "'response', for 'mse' training")

  Y = nodeY.get_inputs()[0]
  X = nodeX.get_inputs()[0]
  
  return tf.reduce_sum((Y - X)**2, name="mse")

def cross_entropy(node_dict):
  """
  """
  try:
    nodeY = node_dict['labels']
    nodeX = node_dict['prediction']
  except AttributeError:
    raise AttributeError("You must define two OutputNodes, named 'prediction' and "
                         "'response', for 'cross_entropy' training")

  Y = nodeY.get_inputs()[0]
  X = nodeX.get_inputs()[0]
  
  Y = tf.squeeze(Y, axis=-1)
  ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,
                                                      logits=X,
                                                      name='cross_entropy')
  return tf.reduce_mean(ce)

def mse_reg(node_dict):
  """
  """
  pass

def elbo(node_dict):
  """
  """
  try:
    node_rec = node_dict['Recognition']
    node_gen = node_dict['Generative']
  except AttributeError:
    raise AttributeError("You must define two InnerNodes, named 'Recognition' and "
                         "'Generative', for 'elbo' training")
    
  nodeY = node_dict['response']
  
  Y = nodeY.get_outputs()[0]
  rec_dist = node_rec.dist
  gen_dist = node_gen.dist
  
  return -( gen_dist.log_prob(Y) + rec_dist.entropy() )