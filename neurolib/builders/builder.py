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

import pydot

from neurolib.encoder.deterministic import DeterministicNNNode
from neurolib.utils.utils import check_name


class Builder(abc.ABC):
  """
  An abstract class representing the Builder type. A Builder object builds
  a single Model by taking the following steps: 
  
  i) adds encoder nodes to the Model graph (MG)
  ii) defines directed links between them representing tensors
  iii) builds a tensorflow graph from the Model graph 
  
  A Builder object MUST implement the method _build()
  """
  def __init__(self, scope, batch_size=None):
    """
    Initialize the builder
    
    Args:
      scope (str): The tensorflow scope of the Model to be built
      
      batch_size (int): The batch size. Defaults to None (unspecified)
    """
    self.scope = scope
    self.batch_size = batch_size
    
    self.num_nodes = 0
    
    # Dictionaries that map name/label to node for the three node types.
    self.nodes = {}
    self.input_nodes = {}
    self.output_nodes = {}
    self._label_to_node = {}

    # The graph of the model
    self.model_graph = pydot.Dot(graph_type='digraph')

  @check_name
  def addInner(self, *main_params, node_class=DeterministicNNNode, name=None,
               **dirs):
    """
    Add an InnerNode to the Encoder Graph
    
    Args:
      *main_params (list): List of mandatory params for the InnerNode
      
      node_class (InnerNode): class of the node
      
      name (str): A unique string identifier for the node being added to the MG
      
      dirs (dict): A dictionary of directives for the node
    """
    label = self.num_nodes
    self.num_nodes += 1
    
    if node_class._requires_builder:
      enc_node = node_class(label, *main_params, name=name,
                            builder=self, 
                            batch_size=self.batch_size,
                            **dirs)
    else:
      enc_node = node_class(label, *main_params, name=name,
                            batch_size=self.batch_size,
                            **dirs)
      
    self.nodes[enc_node.name] = self._label_to_node[label] = enc_node
  
    # Add properties for visualization
    self.model_graph.add_node(enc_node.vis)
    
    return enc_node.name
  
  @abc.abstractmethod
  def _build(self): 
    """
    Build the Model.
    
    Builders MUST implement this method
    """
    raise NotImplementedError("Builders must implement _build")
  
  def visualize_model_graph(self, filename="model_graph"):
    """
    Generate a visual representation of the Model graph
    """
    self.model_graph.write_png(self.scope+filename)