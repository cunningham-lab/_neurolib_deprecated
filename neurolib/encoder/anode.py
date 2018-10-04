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

from neurolib.encoder import *

class ANode(abc.ABC):
  """
  An abstract class for Nodes, the basic building block of the neurolib 
  
  An ANode is an abstraction of an operation, much like tensorflow ops, with
  tensors entering and exiting the node. Compared to tensorflow nodes, ANodes
  are meant to represent higher level abstractions. That is operations that
  broadly correspond to the encoding of input information into some output info.
  ANodes represent relevant stops in the flow of information through a
  statistical model.
  
  Ex: The Variational Autoencoder Model graph is given by: 
  
  [0->d_X] => [d_X->d_Z] => [d_Z->d_X] => [d_X->0]
  
  where the => arrows represent tensors and the [...] represent ANodes.
  [d_X->d_Z] represents an ANode whose input is of dimension d_X and whose
  output is of dimension d_Z, d_Z < d_X. At the ends of the chain there is an
  InputNode and an OutputNode, special types of ANode that represent
  respectively the Model's sources and sinks of information.
  
  ANodes are not meant to be used raw, but rather accessed (built and linked)
  through a Builder object, that in turns belongs to a Model. A Model is a
  directed graph whose nodes are ANodes and whose edges represent conditional
  dependencies between ANodes (see Model docs).
  
  Some ANodes, such as the ConcatNode serve a glue role, stitching together
  other ANodes of the computational graph.
  
  Upon initialization, an ANode holds specifications of its role in the full
  tensorflow graph of a Model to which the ANode belongs. The tensorflow ops are
  not built at initialization, but only when the ANode._build() method is called
  by the Builder object (see Builder docs).
  
  Children of ANode MUST implement the _build() method.
  
  The algorithm to build the tensorflow graph  of the Model depends on 3
  ANode dictionaries that work together:
    self._built_parents : Keeps track of which among the parents of this node
        have been built. A node can only be built once all of its parents have
        been visited
    self._child_to_oslot : The keys are the labels of self's children. For each
        key, the only value value is an integer, the oslot in self that maps to
        that child.
    self._parent_to_islot : The keys are the labels of self's parents, the only
        value is an integer, the islot in self that maps to that child. 
  """
  def __init__(self, label):
    """
    Initializes an ANode
    
    Args:
      label: A unique integer identifier for this node. Typically provided by a
          Builder object.
    
    TODO: The client should be able to pass a tensorflow graph directly. In that
    case, ANode should act as a simple wrapper that returns the input and the
    output.
    """
    self.label = label
  
    self._num_declared_inputs = 0
    self._num_declared_outputs = 0
    
    # Dictionaries for access    
    self._islot_to_shape = {}
    self._oslot_to_shape = {}
    self._islot_to_itensor = {}
    self._oslot_to_otensor = {}
    
    self._built_parents = {}
    self._child_to_oslot = {}
    self._parent_to_islot = {}
    
    self._is_built = False
    
  @property
  def num_inputs(self):
    """
    Return the number of declared inputs.
    
    Useful for cheks and debugging. This number should never change after a node
    is built.
    """
    return self._num_declared_inputs
  
  @num_inputs.setter
  @abstractmethod
  def num_inputs(self, value):
    """
    Setter for num_inputs
    """
    raise NotImplementedError("Please implement me")
  
  @property
  def num_outputs(self):
    """
    Return the number of declared outputs.
    
    Useful for cheks and debugging. This number should never change after a node
    is built.
    """
    return self._num_declared_outputs
  
  @num_outputs.setter
  @abstractmethod
  def num_outputs(self, value):
    """
    Setter for num_oututs
    """
    raise NotImplementedError("Please implement me")

  def get_output_shapes(self):
    """
    Returns a dictionary whose keys are the oslots of the ANode and whose values
    are the outgoing shapes.
    """
    return self._oslot_to_shape
  
  def get_input_shapes(self):
    """
    Returns a dictionary whose keys are the oslots of the ANode and whose values
    are the outgoing shapes.
    """
    return self._islot_to_shape
  
  def get_inputs(self):
    """
    Returns a dictionary whose keys are the islots of the ANode and whose values
    are the incoming tensorflow Tensors.
    
    Requires the node to be built.
    """
    if not self._is_built:
      raise NotImplementedError("A Node must be built before its inputs and "
                                "outputs can be accessed")
    return self._islot_to_itensor
    
  def get_outputs(self):
    """
    Returns a dictionary whose keys are the oslots of the ANode and whose values
    are the outgoing tensorflow Tensors.
    
    Requires the node to be built.
    """
    if not self._is_built:
      raise NotImplementedError("A Node must be built before its inputs and "
                                "outputs can be accessed")
    return self._oslot_to_otensor
    
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")
  
  
