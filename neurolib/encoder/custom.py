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

import pydot

from neurolib.encoder.anode import ANode
from neurolib.encoder.deterministic import DeterministicNNNode

class CustomEncoderNode(ANode):
  """
  TODO: Access to this class through the builder!
  """
  def __init__(self, label, builder, scope=None):
    """
    """
    self._innernode_to_avlble_islots = {}
    self._innernode_to_avlble_oslots = {}
    self._islot_to_enc_islot = {}
    self._oslot_to_enc_oslot = {}
    super(CustomEncoderNode, self).__init__(label)
    
    self._is_committed = False
    self.name = scope
    
    self._builder = builder
    
    # Add visualization
    self.vis = pydot.Node(self.name)

    
  @ANode.num_declared_inputs.setter
  def num_declared_inputs(self, value):
    """
    """
    if self._is_committed:
      if value > self.num_expected_inputs:
        raise AttributeError("Attribute num_inputs of this CustomNode cannot exceed ",
                             self.num_expected_inputs)
    else:
      self._num_declared_inputs = value

  @ANode.num_declared_outputs.setter
  def num_declared_outputs(self, value):
    """
    """
    if self._is_committed:
      if value != self.num_expected_outputs:
        raise AttributeError("Attribute num_outputs of this CustomNode is set to ",
                             self.num_expected_outputs)
    else:
      self._num_declared_outputs = value

  def addInner(self, *main_params, name=None, node_class=DeterministicNNNode,
               directives={}):
    """
    """
    node_label = self._builder.addInner(*main_params, name=name,
                                        node_class=node_class,
                                        directives=directives)
    node = self._builder.nodes[node_label]
    
    # Assumes fixed number of expected_inputs
    self._innernode_to_avlble_islots[node_label] = list(
                                    range(node.num_expected_inputs))
    self._innernode_to_avlble_oslots[node_label] = list(
                                    range(node.num_expected_outputs))
    
    return node.label
    
  def addDirectedLink(self, enc1, enc2, islot=0, oslot=0):
    """
    """
    if isinstance(enc1, int):
      enc1 = self._builder.nodes[enc1]
    if isinstance(enc2, int):
      enc2 = self._builder.nodes[enc2]
    self._builder.addDirectedLink(enc1, enc2, islot, oslot)
    
    # Remove the connected islot and oslot from the lists of available ones
    self._innernode_to_avlble_oslots[enc1.label].remove(oslot)
    self._innernode_to_avlble_islots[enc2.label].remove(enc2.num_inputs-1)
    
  def commit(self):
    """
    Prepare the CustomNode for building.
    
    A) 
    
    B)
    """
    print('BEGIN COMMIT')
    # Stage A
    self.num_expected_inputs = 0
    for node_label, islot_list in self._innernode_to_avlble_islots.items():
      if islot_list:
        node = self._builder.nodes[node_label]
        self._builder.input_nodes[node.label] = node
#         for i in range(len(islot_list)):
#           self._islot_to_enc_islot[self.num_expected_inputs] = (node, islot_list[i])
#           self.num_expected_inputs += 1
#         for i in range(len(islot_list)):
        for islot in islot_list:
          self._islot_to_enc_islot[self.num_expected_inputs] = (node, islot)
          self.num_expected_inputs += 1
        
    # Stage B
    print('self._innernode_to_avlble_oslots', self._innernode_to_avlble_oslots)
    for node_label, oslot_list in self._innernode_to_avlble_oslots.items():
      print('oslot_list', oslot_list, bool(oslot_list))
      if oslot_list:
        node = self._builder.nodes[node_label]
        self._builder.output_nodes[node.label] = node
        print('node.get_output_shapes()', node.get_output_shapes())
        for i in range(len(oslot_list)):
          self._oslot_to_shape[self.num_outputs] = node.get_output_shapes()[oslot_list[i]]
          self._oslot_to_enc_oslot[self.num_outputs] = (node, oslot_list[i])
          self.num_outputs += 1
      print('self._oslot_to_enc_oslot', self._oslot_to_enc_oslot)
      print('self.num_outputs', self.num_outputs)
    self.num_expected_outputs = self.num_outputs
    
    self._is_committed = True
    print('END COMMIT')
      
  def get_output_shapes(self):
    """
    """
    if self._is_committed:
      return ANode.get_output_shapes(self)
    else:
      raise NotImplementedError("The CustomNode is not committed yet. Please "
                                "commit in order to gain access to the output "
                                "shapes")
      
  def _build(self):
    """
    """
    print("BEGIN BUILD")
    self._builder._build()
    
    print('num_outputs', self.num_outputs)
    for i in range(self.num_outputs):
      onode, oslot = self._oslot_to_enc_oslot[i]
      self._oslot_to_otensor[i] = onode.get_outputs()[oslot]
    print('self._oslot_to_otensor', self._oslot_to_otensor)
        
      
      # TODO: need to deal with what happens upon exit 
    
    self._is_built = True
    print("END BUILD")
    
  def get_node(self, label):
    """
    """
    return self._builder.nodes[label]
    