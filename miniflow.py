import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # New property! Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        self.value = None

    def forward(self):
        """
        Forward propagation.
        
        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)
    
    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Add(Node):
    def __init__(self, *inNodes):
        Node.__init__(self, list(inNodes))
    
    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """
        self.value = 0
        for i in self.inbound_nodes:
            self.value += i.value

class Mul(Node):
    def __init__(self, *inNodes):
        Node.__init__(self, list(inNodes))
    
    def forward(self):
        if len(self.inbound_nodes[0]) == 0:
            return
        
        if not (len(self.inbound_nodes[0]) == len(self.inbound_nodes[1])):
            return

        self.value = np.ones(len(self.inbound_nodes[0]))
        for i in self.inbound_nodes:
            self.value = np.multiply(self.value,i.value)

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
    
    # NOTE: The weights and bias properties here are not
    # numbers, but rather references to other nodes.
    # The weight and bias values are stored within the
    # respective nodes.
    
    def forward(self):
        """
            Set self.value to the value of the linear function output.
            
            Your code goes here!
            """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        
        self.value = np.dot(X,W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """
        You need to fix the `_sigmoid` and `forward` methods.
        """
    def __init__(self, node):
        Node.__init__(self, [node])
    
    def _sigmoid(self, x):
        """
            This method is separate from `forward` because it
            will be used later with `backward` as well.
            
            `x`: A numpy array-like object.
            
            Return the result of the sigmoid function.
            
            Your code here!
            """
        return 1/(1+np.exp(-x))
    
    def forward(self):
        """
            Set the value of this node to the result of the
            sigmoid function, `_sigmoid`.
            
            Your code here!
            """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
                
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            """
            TODO: Your code goes here!
            Set the gradients property to the gradients with respect to each input.
            NOTE: See the Linear node and MSE node for examples.
            """
            sigmoid_val = self._sigmoid(self.inbound_nodes[0].value)
            part_deriv = sigmoid_val*(1-sigmoid_val)
            
            self.gradients[self.inbound_nodes[0]] += np.multiply(part_deriv,grad_cost )

class MSE(Node):
    def __init__(self, y, a):
        """
            The mean squared error cost function.
            Should be used as the last node for a network.
            """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])
    
    def forward(self):
        """
            Calculates the mean squared error.
            """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        # TODO: your code here
        self.m = self.inbound_nodes[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(np.square(self.diff))

    def backward(self):
        """
        Calculates the gradient of the cost.
        
        This is the final node of the network so outbound nodes
        are not a concern.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

def topological_sort(feed_dict):
    """
        Sort generic nodes in topological order using Kahn's Algorithm.
        
        `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
        
        Returns a list of sorted nodes.
        """
    
    input_nodes = [n for n in feed_dict.keys()]
    
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        
        if isinstance(n, Input):
            n.value = feed_dict[n]
        
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
        Performs a forward pass through a list of sorted nodes.
        
        Arguments:
        
        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.
        
        Returns the output Node's value
        """

    for n in sorted_nodes:
        n.forward()

    return output_node.value

def forward_and_backward(graph):
    """
        Performs a forward pass and a backward pass through a list of sorted nodes.
        
        Arguments:
        
        `graph`: The result of calling `topological_sort`.
        """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()
