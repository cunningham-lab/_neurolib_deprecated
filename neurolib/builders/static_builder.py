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
import tensorflow as tf

from neurolib.builders.builder import Builder
from neurolib.encoder.deterministic import DeterministicNNNode
from neurolib.encoder.anode import ANode
from neurolib.encoder.custom import CustomNode
from neurolib.encoder.input import PlaceholderInputNode  # @UnusedImport
from neurolib.encoder.output import OutputNode
from neurolib.utils.utils import check_name

# pylint: disable=bad-indentation, no-member, protected-access

class StaticBuilder(Builder):
  """
  A StaticBuilder is a Builder for statistical models or nodes that do not
  involve sequential data. In particular, models of time series cannot be built
  using a StaticBuilder.
  
  Building of a static Model through a StaticBuilder is done in two stages:
  Declaration and Construction. In the Declaration stage, the input, output and
  inner nodes of the Model are 'added' to the Model graph (MG), and directed
  links - representing the flow of tensors - are defined between them. In the
  Construction stage, a BFS-like algorithm is called that generates a tensorflow
  graph out of the MG specification
  
  A StaticBuilder defines the following key methods
  
    addOutput(): ...
    
    addInput(): ...
    
    addDirectedLink(): ...
    
    build()
    
  Ex: The following code builds a simple regression Model
      
      builder = StaticBuilder(scope='regression')
      in0 = builder.addInput(input_dim, name="features")
      enc1 = builder.addInner(output_dim, **dirs)
      out0 = builder.addOutput(name="prediction")
      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)

      in1 = builder.addInput(output_dim, name="input_response")
      out1 = builder.addOutput(name="response")
      builder.addDirectedLink(in1, out1)
      
      builder.build()
    
    The 2 input nodes define placeholders for the features and response data
  
  """
  def __init__(self, scope=None, batch_size=None):
    """
    Initialize the StaticBuilder
    
    Args:
      scope (str): The tensorflow scope of the Model to be built
      batch_size (int): The batch size. Defaults to None (unspecified)
    """
    super(StaticBuilder, self).__init__(scope, batch_size=batch_size)
    
    self.custom_encoders = {}
    
    self.adj_matrix = None
    self.adj_list = None
              
  @check_name
  def addInput(self, *main_params,
               name=None,
               iclass=PlaceholderInputNode,
               **dirs):
    """
    Add an InputNode to the Encoder Graph
    
    Args:
      *main_params (list): Mandatory parameters for the InputNode
      name (str): Unique identifier for the Input Node
      iclass (InputNode): class of the node
      dirs (dict): A dictionary of directives for the node
      
    TODO: Do not call class names directly
    """
    label = self.num_nodes
    self.num_nodes += 1

    if iclass._requires_builder:
      in_node = iclass(label, *main_params,
                       batch_size=self.batch_size,
                       name=name,
                       builder=self,
                       **dirs)
    else:
      in_node = iclass(label, *main_params,
                       batch_size=self.batch_size,
                       name=name,
                       **dirs)

    name = in_node.name
    self.input_nodes[name] = self.nodes[name] = in_node 
    self._label_to_node[label] = in_node
    
    # Add properties for visualization
    self.model_graph.add_node(in_node.vis)

    return name
    
  @check_name
  def addOutput(self, name=None):
    """
    Add an OutputNode to the Encoder Graph
    
    Args:
      name (str): Unique identifier for the Output Node
    """
    label = self.num_nodes
    self.num_nodes += 1
    out_node = OutputNode(label, name=name)
    name = out_node.name
    self.output_nodes[name] = self.nodes[name] = out_node 
    self._label_to_node[label] = out_node
    
    # Add properties for visualization
    self.model_graph.add_node(out_node.vis)

    return name
    
  def addDirectedLink(self, node1, node2, oslot=0, islot=0):
    """
    Add directed links to the Encoder graph. 
    
    A) Deal with different item types. The client may provide as arguments,
    either EncoderNodes or integers. Get the EncoderNodes in the latter case
 
    B) Check that the provided oslot for node1 is free. Otherwise, raise an
    exception.
    
    C) Initialize/Add dimensions the graph representations stored in the
    builder. Specifically, the first time a DirectedLink is added an adjacency
    matrix and an adjacency list are created. From then on, the appropriate
    number of dimensions are added to these representations.

    D) Update the representations to represent the new link. 
    
    E) Fill the all important dictionaries _child_to_oslot and _parent_to_islot.
    For node._child_to_oslot[key] = value, key represents the labels of the
    children of node, while the values are the indices of the oslot in node
    that outputs to that child. Analogously, in node._parent_to_islot[key] =
    value, the keys are the labels of the parents of node and the values are the
    input slots in node corresponding to each key.
    
    F) Possibly update the attributes of node2. In particular deal with nodes
    whose output shapes are dynamically inferred. This is important for nodes such
    as CloneNode and ConcatNode whose output shapes are not provided at
    creation. Once these nodes gather their inputs, they can infer their
    output_shape at this stage.
    
    Args:
      node1 (ANode): Node from which the edge emanates
      node2 (ANode): Node to which the edge arrives
      oslot (int): Output slot in node1
      islot (int): Input slot in node2
    """
    # A
    if isinstance(node1, str):
      node1 = self.nodes[node1]
    if isinstance(node2, str):
      node2 = self.nodes[node2]
    if not (isinstance(node1, ANode) and isinstance(node2, ANode)):
      raise TypeError("Args node1 and node2 must be either of type `str` "
                      "or type `ANode`")
    
    # B
    nnodes = self.num_nodes
    if not node1._oslot_to_shape:
      if isinstance(node1, OutputNode):
        raise ValueError("Outgoing directed links cannot be defined for "
                         "OutputNodes")
      else:
        raise ValueError("Node1 appears to have no outputs. This software has "
                         "no clue why that would be.\n Please report to my "
                         "master.")
    elif oslot not in node1._oslot_to_shape:
      raise KeyError("The requested oslot has not been found. Inferring this "
                     "oslot shape may require knowledge of the shape of its "
                     "inputs. In that case, all the inputs for this node must "
                     "be declared")
    if islot in node2._islot_to_shape:
      raise AttributeError("That input slot is already occupied. Assign to "
                           "a different islot")

    # C
    print('Adding dlink', node1.label, ' -> ', node2.label)
    if self.adj_matrix is None:
      self.adj_matrix = [[0]*nnodes for _ in range(nnodes)]
      self.adj_list = [[] for _ in range(nnodes)]
    else:
      if nnodes > len(self.adj_matrix):
        l = len(self.adj_matrix)
        for row in range(l):
          self.adj_matrix[row].extend([0]*(nnodes-l))
        for _ in range(nnodes-l):
          self.adj_matrix.append([0]*nnodes)
          self.adj_list.append([])
    
    # D
    self.adj_matrix[node1.label][node2.label] = 1
    self.adj_list[node1.label].append(node2.label)
#     print('After:', self.adj_list)
    self.model_graph.add_edge(pydot.Edge(node1.vis, node2.vis))
      
    # E
    if node1.num_expected_outputs > 1:
      if oslot is None:
        raise ValueError("The in-node has more than one output slot, so pairing "
                         "to the out-node is ambiguous.\n You must specify the "
                         "output slot. The declared output slots for node 1 are: ",
                         node1._oslot_to_shape)
    if node2.num_expected_inputs > 1:
      if islot is None:
        raise ValueError("The out-node has more than one input slot, so pairing "
                         "from the in-node is ambiguous.\n You must specify the " 
                         "input slot")
    exchanged_shape = node1._oslot_to_shape[oslot]
    node1._child_label_to_oslot[node2.label] = oslot
    if oslot in node1.free_oslots:
      node1.num_declared_outputs += 1
      node1.free_oslots.remove(oslot)
    
    node2._islot_to_shape[islot] = exchanged_shape
    node2._parent_label_to_islot[node1.label] = islot    
    node2.num_declared_inputs += 1
    
    # F
    update = getattr(node2, '_update_when_linked_as_node2', None)
    if callable(update):
      node2._update_when_linked_as_node2()

    # Initialize _built_parents for the child node. This is used in the build
    # algorithm below.
    node2._built_parents[node1.label] = False
      
  def check_graph_correctness(self):
    """
    Checks the graph declared so far. 
    
    TODO:
    """
    pass
        
  def createCustomNode(self,
                       num_inputs,
                       num_outputs,
                       name=None):
    """
    Create a custom node
    
    TODO:
    """
    label = self.num_nodes
    self.num_nodes += 1
    
    # Define here to avoid circular dependencies
    custom_builder = StaticBuilder(scope=name, batch_size=self.batch_size)
    cust = CustomNode(label,
                      num_inputs,
                      num_outputs,
                      builder=custom_builder,
                      name=name)
    self.custom_encoders[name] = self.nodes[label] = cust
    self._label_to_node[label] = cust
    return cust
  
  def get_custom_encoder(self, name):
    """
    Get a CustomNode by name
    """
    return self.custom_encoders[name] 
  
  def add_to_custom(self,
                    custom_node,
                    output_shapes,
                    name=None,
                    node_class=DeterministicNNNode,
                    **dirs):
    """
    Add an InnerNode to a CustomNode
    """
    custom_node.builder.addInner(output_shapes,
                                 name=name,
                                 node_class=node_class,
                                 **dirs)

  def get_label_from_name(self, name):
    """
    Get the label of a node from name
    """
    return self.nodes[name].label

  def build(self):
    """
    Build the declared model.
    
    # put all nodes in a waiting list of nodes
    # for node in input_nodes:
      # start BFS from node. Add node to queue.
      # (*)Dequeue, mark as visited
      # build the tensorflow graph with the new added node
      # Look at all its children nodes.
      # For child in children of node
          Add node to the list of inputs of child
      #   have we visited all the parents of child?
          Yes
            Add to the queue
      # Go back to (*)
      * If the queue is empty, exit, start over from the next input node until all 
      # have been exhausted
      
      # TODO: deal with back links.
    """       
    self.check_graph_correctness()
    
    print('\nBEGIN MAIN BUILD')
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE): 
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        
        # start BFS
        queue.append(cur_inode_label)
        while queue:
          # A node is visited by definition once it is popped from the queue
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
  
          print("Building node: ", cur_node.label, cur_node.name)
          # Build the tensorflow graph for this Encoder
          cur_node._build()
                    
          # Go over the current node's children
          for child_label in self.adj_list[cur_node_label]:
            child_node = self._label_to_node[child_label]
            child_node._built_parents[cur_node_label] = True
            
            oslot = cur_node._child_label_to_oslot[child_label]
            islot = child_node._parent_label_to_islot[cur_node_label]
            
            # Fill the inputs of the child node
#             print('cur_node', cur_node_label, cur_node.name)
#             print('cur_node.get_outputs()', cur_node.get_outputs() )
            child_node._islot_to_itensor[islot] = cur_node.get_outputs()[oslot]
            if isinstance(child_node, CustomNode):
              enc, enc_islot = child_node._islot_to_inner_node_islot[islot]
              enc._islot_to_itensor[enc_islot] = cur_node.get_outputs()[oslot]
            
            # If the child is an OutputNode, we can append to the queue right away
            # (OutputNodes have only one input)
            if isinstance(child_node, OutputNode):
              queue.append(child_node.label)
              continue
            
            # A child only gets added to the queue, i.e. ready to be built, once
            # all its parents have been built ( and hence, produced the
            # necessary inputs )
            if all(child_node._built_parents.items()):
              queue.append(child_node.label)
    
    print('END MAIN BUILD')  