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

from neurolib.encoder.basic import InnerNode
from neurolib.encoder.deterministic import DeterministicNNNode  # @UnusedImport
from neurolib.encoder.input import PlaceholderInputNode

# pylint: disable=bad-indentation, no-member, protected-access


class CustomNode(InnerNode):
  """
  A CustomNode is an InnerNode implementing representing an arbitrary map, taking an
  arbitrary number of inputs and returning an arbitrary number of outputs.
  CustomNodes may be used to put together portions of the computational graph
  into an encoder that can be subsequently treated as a black box.
  """
  def __init__(self,
               out_builder,
               in_builder,
               num_inputs,
               num_outputs,
               is_sequence=False,
               name=None):
    """
    Initialize a CustomNode
    
    _islot_to_inner_node_islot ~ {islot : (inner_node, inner_node_islot)}
    _oslot_to_inner_node_oslot ~ {oslot : (inner_node, inner_node_oslot)}

    """
    super(CustomNode, self).__init__(out_builder,
                                     is_sequence)
    
    if num_inputs is None:
      raise NotImplementedError("CustomNodes with unspecified number of inputs"
                                "not implemented")
    if num_outputs is None:
      raise NotImplementedError("CustomNodes with unspecified number of outputs"
                                "not implemented")
    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs

    self.in_builder = in_builder
    self.name = 'Cust_' + str(self.label) if name is None else name

    self._innernode_to_its_avlble_islots = {}
    self._innernode_to_its_avlble_oslots = {}
    self._islot_to_inner_node_islot = {}
    self._oslot_to_inner_node_oslot = {}
    
    self._is_committed = False
    self._is_built = False
    
    self.free_oslots = list(range(self.num_expected_outputs))

  def addInput(self, islot=None, inode_name=None, inode_islot=None):
    """
    Declare an inner islot as an input to the CustomNode
    """
    if any(elem is None for elem in [islot, inode_name, inode_islot]):
      raise ValueError("Missing argument")
    self._islot_to_inner_node_islot[islot] = (inode_name, inode_islot) 
  
  def addOutput(self, oslot=None, inode_name=None, inode_oslot=None):
    """
    Declare an inner oslot as an output to the CustomNode 
    """
    if any(elem is None for elem in [oslot, inode_name, inode_oslot]):
      raise ValueError("Missing argument")

    self._oslot_to_inner_node_oslot[oslot] = (inode_name, inode_oslot)
    
    inode = self.in_builder.nodes[inode_name]
    self._oslot_to_shape[oslot] = inode.get_oslot_shape(inode_oslot)
    
  def addInner(self, 
               *main_params,
               name=None,
               node_class=DeterministicNNNode,
               **dirs):
    """
    Add an InnerNode to the CustomNode
    """
    node_name = self.in_builder.addInner(*main_params,
                                         name=name,
                                         node_class=node_class,
                                         **dirs)
    node = self.in_builder.nodes[node_name]
    
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
      enc1 = self.in_builder.nodes[enc1]
    if isinstance(enc2, str):
      enc2 = self.in_builder.nodes[enc2]
    
    self.in_builder.addDirectedLink(enc1, enc2, islot, oslot)
    # Remove the connected islot and oslot from the lists of available ones
    self._innernode_to_its_avlble_oslots[enc1.name].remove(oslot)
    self._innernode_to_its_avlble_islots[enc2.name].remove(islot)
    
  def commit(self):
    """
    Prepare the CustomNode for building.
    
    In particular fill, the dictionaries 
      _islot_to_inner_node_islot ~ {inode_islot : (inner_node, inner_node_islot)}
      _oslot_to_inner_node_oslot ~ {inode_oslot : (inner_node, inner_node_oslot)}
  
    Stage A: If an InnerNode of the CustomNode has an available inode_islot, then this
    is an input to the CustomNode.
    """
    print('BEGIN COMMIT')
    # Stage A
    assigned_islots = 0
    for inode_name, islot_list in self._innernode_to_its_avlble_islots.items():
      if islot_list:
        inode = self.in_builder.nodes[inode_name]
        self.in_builder.input_nodes[inode.name] = inode
        for inode_islot in islot_list:
          if (inode_name, inode_islot) not in self._islot_to_inner_node_islot.values():
            print("(inode_name, inode_islot)", (inode_name, inode_islot))
            print("self._islot_to_inner_node_islot.items()",
                  self._islot_to_inner_node_islot.values())
            raise ValueError("")
          assigned_islots += 1
    if assigned_islots != self.num_expected_inputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
        
#     Stage B
    assigned_oslots = 0
    for inode_name, oslot_list in self._innernode_to_its_avlble_oslots.items():
      if oslot_list:
        inode = self.in_builder.nodes[inode_name]
        self.in_builder.output_nodes[inode.name] = inode
        for inode_oslot in oslot_list:
          print("(inode_name, inode_islot)", (inode_name, inode_oslot))
          print("self._islot_to_inner_node_islot.items()",
                self._oslot_to_inner_node_oslot.values())
          print((inode_name, inode_oslot) in self._oslot_to_inner_node_oslot.values())
          if (inode_name, inode_oslot) not in self._oslot_to_inner_node_oslot.values():
            raise ValueError("")
          assigned_oslots += 1
    if assigned_oslots != self.num_expected_outputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
    
    self._is_committed = True
    print('END COMMIT')

  def _build(self, islot_to_itensor=None):
    """
    Build the CustomNode
    """
    print("islot_to_itensor", islot_to_itensor)
    if islot_to_itensor is None:
      islot_to_itensor = self._islot_to_itensor
    # TODO: This needs to be done differently for RNN cells! There is no concept
    # of "being built" for RNNCells
    
#     if not self._is_built or islot_to_itensor is None:
    print("\nBEGIN CUSTOM BUILD")
#     islot_to_itensor = self._islot_to_itensor

    num_outputs = self.num_expected_outputs
    if self.num_declared_outputs != num_outputs:
      raise ValueError("`self.num_declared_outputs != self.num_expected_outputs`",
                       (self.num_declared_outputs, self.num_expected_outputs))
    
    self.in_builder.build()
    for i in range(num_outputs):
      onode_name, oslot = self._oslot_to_inner_node_oslot[i]
      onode = self.in_builder.nodes[onode_name]
      self._oslot_to_otensor[i] = onode.get_outputs()[oslot]
    
    output = self._oslot_to_otensor
    self._is_built = True
    print("END CUSTOM BUILD\n")
#     else:
#       print("\nBEGIN CUSTOM BUILD")
#       temp = self._islot_to_itensor # horrible hack to allow calling for a CustomNode
#       self._islot_to_itensor = islot_to_itensor
#       
#       self.in_builder.build()
#       output = {}
#       for i in range(num_outputs):
#         onode_name, oslot = self._oslot_to_inner_node_oslot[i]
#         onode = self.in_builder.nodes[onode_name]
#         output[i] = onode.get_outputs()[oslot]
#       self._islot_to_itensor = temp

    return list(zip(*sorted(output.items())))[1]
    
  def get_node(self, label):
    """
    """
    return self.in_builder.nodes[label]