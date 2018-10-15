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

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode
    

class InnerNode(ANode):
  """
  An InnerNode is a node that is not an OutputNode nor an InputNode. InnerNodes
  have num_inputs > 0 and num_outputs > 0. Its outputs can be deterministic, as
  in the DeterministicNNNode, or stochastic, as in the NormalTriLNode. 
  
  Some InnerNodes are utility nodes, their purpose being to stitch together
  other InnerNodes into the Model graph (MG). This is the case for instance of
  the MergeConcatNode and the CloneNode.  
  """
  def __init__(self, label):
    """
    Initialize the InnerNode
    
    Args:
      label (int): A unique integer identifier for the node
    """
    super(InnerNode, self).__init__(label)
              
    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')
      


    