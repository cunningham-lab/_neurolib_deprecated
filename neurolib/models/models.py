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
from abc import abstractmethod

# pylint: disable=bad-indentation, no-member, protected-access

class Model(abc.ABC):
  """
  An abstract class for Machine Learning Models.
  
  Classes that inherit from the abstract class Model will be seen by the client.
  Models are created through a Builder object, which in turn is endowed with an
  interface via which the client may add Nodes and Links to the Model as
  desired.
  
  There are two ways of invoking a Model.
  
  a) Default
    An instance of Model is created using a set of mandatory directives specific
    to the Model as in
      model = Model(*args, **dirs)
    Some control over the model hyperparameters is provided through the
    directives `**dirs`. The Model Builder object is automatically created.
    
  b) Custom
    The instance of Model is created through a custom Builder object
      model = Model(builder=mybuilder)
    where the custom builder object is used to declare the graphical model as
    desired by the user. This provides a lot more control over the model
    hyperparameters since each node of the Model graph can be independently
    tweaked.
    
  The Model classes should implement at the very least the following methods
  
  train(...)
  sample(...)
  """
  def __init__(self):
    """
    TODO: Should I start a session here? This presents some troubles right now,
    at least with this implementation of get_session which I am beginning to
    suspect it is not going to cut it for our purposes. The sessions needs to be
    micromanaged...
    
    TODO: I also want to manually manage the graphs for when people want to run
    two models and compare for example.
    """
    self.inputs = {}
    self.outputs = {}
    self._is_built = False
    
  @property
  def main_scope(self):
    """
    """
    return self._main_scope
      
  @abstractmethod
  def build(self):
    """
    """
    raise NotImplementedError("")
        
  @abstractmethod
  def update(self, dataset):
    """
    """
    raise NotImplementedError("")

  @abstractmethod
  def train(self, dataset, **kwargs):
    """
    """
    raise NotImplementedError("")
  
  def sample(self):
    """
    TODO: Think about what is meant by sample. Only Generative Models should
    have sampling capabilities so this function probably shouldnt be here since
    I am defining a `Model` as the most general composition of Encoders. sample
    in fact seems to be a method of the abstract Encoder class, not of the
    Model, huh?
    
    TODO: Fill the exception
    """
    raise NotImplementedError("")

  def make_datasets(self, dataset):
    """
    Splits the dataset dictionary into train, validation and test datasets.
    """
    scope = self.main_scope
    train_dataset = {}
    valid_dataset = {}
    test_dataset = {}
    for key in dataset:
      d_set, inode = key.split('_')[0], key.split('_')[-1]
      if d_set == 'train':
        train_dataset[scope + '/' + inode + ':0'] = dataset[key]
      elif d_set == 'valid':
        valid_dataset[key] = dataset[key]
      elif d_set == 'test':
        test_dataset[key] = dataset[key]
      else:
        raise KeyError("The dataset contains the key `{}`. The only allowed "
                       "prefixes for keys in the dataset are 'train', "
                       "'valid' and 'test'".format(key))
    
    return train_dataset, valid_dataset, test_dataset
  

class StaticModel(Model):
  """
  TODO: Decide if this split is needed
  """
  def __init__(self):
    """
    """
    super(StaticModel, self).__init__()


class SequentialModel():
  """
  TODO: Decide if this split is needed
  """
  def __init__(self):
    """
    """
    super(SequentialModel, self).__init__()
  
