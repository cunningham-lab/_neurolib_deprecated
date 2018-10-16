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
import pydot

from neurolib.encoder.anode import ANode
from neurolib.encoder.deterministic import DeterministicNNNode

# pylint: disable=bad-indentation, no-member, protected-access

class CustomNode(ANode):
  """
  TODO: Access to this class through the builder!
  """
  def __init__(self, label,
               num_inputs,
               num_outputs,
               builder,
               name=None):
    """
    Initialize a CustomNode
    """
    if num_inputs is None:
      raise NotImplementedError("CustomNodes with unspecified number of inputs"
                                "not implemented")
    if num_outputs is None:
      raise NotImplementedError("CustomNodes with unspecified number of outputs"
                                "not implemented")
    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs
    super(CustomNode, self).__init__(label)
    
    self._builder = builder
    
    self._innernode_to_its_avlble_islots = {}
    self._innernode_to_its_avlble_oslots = {}
    self._islot_to_inner_node_islot = {}
    self._oslot_to_inner_node_oslot = {}
    
    self.name = 'Cust_' + self.label if name is None else name
        
    self._is_committed = False
    self._is_built = False
    
    self.free_oslots = list(range(self.num_expected_outputs))
    print("self.free_oslots", self.free_oslots)
    
    # Add visualization
    self.vis = pydot.Node(self.name)

    
  @ANode.num_declared_inputs.setter
  def num_declared_inputs(self, value):
    """
    Setter for num_declared_inputs
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_declared_inputs cannot exceed "
                           "`self.num_expected_inputs`",
                           self.num_expected_inputs)
    self._num_declared_inputs = value

  @ANode.num_declared_outputs.setter
  def num_declared_outputs(self, value):
    """
    Setter for num_declared_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_declared_outputs cannot exceed "
                           "`self.num_expected_outputs`", 
                           self.num_expected_outputs)
    self._num_declared_outputs = value

  def addInner(self, 
               *main_params,
               name=None,
               node_class=DeterministicNNNode,
               **dirs):
    """
    Add an InnerNode to the CustomNode
    """
    node_name = self._builder.addInner(*main_params,
                                        name=name,
                                        node_class=node_class,
                                        **dirs)
    node = self._builder.nodes[node_name]
    
    # Assumes fixed number of expected_inputs
    self._innernode_to_its_avlble_islots[node_name] = list(
                                    range(node.num_expected_inputs))
    self._innernode_to_its_avlble_oslots[node_name] = list(
                                    range(node.num_expected_outputs))
    return node.name
    
  def addDirectedLink(self, enc1, enc2, islot=0, oslot=0):
    """
    Add a DirectedLink to the CustomNode inner graph
    """
    if isinstance(enc1, str):
      enc1 = self._builder.nodes[enc1]
    if isinstance(enc2, str):
      enc2 = self._builder.nodes[enc2]
    self._builder.addDirectedLink(enc1, enc2, islot, oslot)
    
    # Remove the connected islot and oslot from the lists of available ones
    self._innernode_to_its_avlble_oslots[enc1.name].remove(oslot)
    self._innernode_to_its_avlble_islots[enc2.name].remove(islot)
    
  def commit(self):
    """
    Prepare the CustomNode for building.
    
    In particular fill, the dictionaries 
      _islot_to_inner_node_islot ~ {islot : (inner_node, inner_node_islot)}
      _oslot_to_inner_node_oslot ~ {oslot : (inner_node, inner_node_oslot)}
    that allow the build algorithm to connect nodes outside the CustomNode to
    its islots and oslots of 
    """
    print('BEGIN COMMIT')
    # Stage A
    assigned_islots = 0
    for node_name, islot_list in self._innernode_to_its_avlble_islots.items():
      if islot_list:
        node = self._builder.nodes[node_name]
        self._builder.input_nodes[node.name] = node
        for islot in islot_list:
          self._islot_to_inner_node_islot[assigned_islots] = (node, islot)
          assigned_islots += 1
    if assigned_islots != self.num_expected_inputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
        
    # Stage B
    assigned_oslots = 0
    for node_name, oslot_list in self._innernode_to_its_avlble_oslots.items():
      if oslot_list:
        node = self._builder.nodes[node_name]
        self._builder.output_nodes[node.name] = node
        for oslot in oslot_list:
          self._oslot_to_shape[assigned_oslots] = node.get_oslot_shape(oslot)
          self._oslot_to_inner_node_oslot[assigned_oslots] = (node, oslot)
          assigned_oslots += 1
    if assigned_oslots != self.num_expected_outputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
    
    self._is_committed = True
    print('END COMMIT')
      
  def _build(self):
    """
    Build the CustomNode
    """
    print("BEGIN CUSTOM BUILD")
    num_outputs = self.num_expected_outputs
    print("self.num_declared_outputs != self.num_expected_outputs:",
          self.num_declared_outputs, self.num_expected_outputs)
    if self.num_declared_outputs != num_outputs:
      raise ValueError("`self.num_declared_outputs != self.num_expected_outputs`")
    
    self._builder.build()
    for i in range(num_outputs):
      onode, oslot = self._oslot_to_inner_node_oslot[i]
      self._oslot_to_otensor[i] = onode.get_outputs()[oslot]
#     print('self._oslot_to_otensor', self._oslot_to_otensor)
            
    self._is_built = True
    print("END CUSTOM BUILD")
    
  def get_node(self, label):
    """
    """
    return self._builder.nodes[label]
    