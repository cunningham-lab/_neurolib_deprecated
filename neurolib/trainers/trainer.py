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
import abc

import numpy as np
import tensorflow as tf

from neurolib.utils.graphs import get_session

# pylint: disable=bad-indentation, no-member, protected-access

def make_data_iterator(data, batch_size=1, shuffle=True):
    """
    """
    if batch_size is None:
      batch_size = 1
    nsamps = len(data[0])
    l_inds = np.arange(nsamps)
    if shuffle: 
        np.random.shuffle(l_inds)
    
    for i in range(0, nsamps, batch_size):
        yield [ d[l_inds[i:i+batch_size]] for d in data ]
            
            
class ModelBender(abc.ABC):
  """
  TODO: Implement training with tensorflow Queues. This is IMPORTANT! Get rid of
  the feed_dict!
  
  TODO: Put the abc functionality to use
  """
  @abc.abstractmethod 
  def update(self):
    """
    """
    raise NotImplementedError("")


class GDTrainer(ModelBender):
  """
  TODO: This class will probably move to a file of its own. We probably want to
  leave this file only for the abstract class.
  """
  opt_dict = {'adam' : tf.train.AdamOptimizer,
              'adagrad' : tf.train.AdagradOptimizer,
              'momentum' : tf.train.MomentumOptimizer,
              'gd' : tf.train.GradientDescentOptimizer}

  def __init__(self,
               cost,
               **dirs):
    """
    Initialize a GDTrainer
    """
    self.cost = cost
    self._update_default_directives(**dirs)
    
    self._define_train_op()
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives.
    """
    self.directives = {'optimizer' : 'adam',
                       'lr' : 1e-4}
    self.directives.update(dirs)

  def _define_train_op(self):
    """
    Define the train op using tensorflow standard machinery.
    """
    directives = self.directives
    optimizer_class = self.opt_dict[directives['optimizer']]
    opt = optimizer_class(self.directives['lr'])
    
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)
    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tf.get_variable_scope().name)
    print('Scope', tf.get_variable_scope().name)
    for i in range(len(self.train_vars)):
        shape = self.train_vars[i].get_shape().as_list()
        print("    ", i, self.train_vars[i].name, shape)

    gradsvars = opt.compute_gradients(self.cost, self.train_vars)
    self.train_op = opt.apply_gradients(gradsvars, global_step=self.train_step,
                                        name='train1_op')
    
  def update(self, sess, keys_and_data, batch_size=1):
    """
    Perform a single gradient descent update for the variables in this cost.
    
    TODO: Get rid of the feed_dict in favor of tensorflow Queues! Add
    multithreading capabilities
    """
    keys, data = keys_and_data
    data_iterator = make_data_iterator(data, batch_size=batch_size)
    for batch in data_iterator:
      feed_dict = dict(zip(keys, batch))
      sess.run([self.train_op], feed_dict=feed_dict)
  
#   def train(self, train_dataset, valid_dataset={}, num_epochs=100):
#     """
#     
#     
#     User is in charge of splitting the dataset into train, validation, etc.
#     """      
#     sess = get_session()
#     sess.run(tf.global_variables_initializer())
#     for _ in range(num_epochs):
#       loss = self.update(sess, tuple(zip(*train_dataset.items()) ) )
#       print(loss)
#       
#     sess.close()
    
    
