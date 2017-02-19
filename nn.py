"""
    This script builds and runs a graph with miniflow.
    
    There is no need to change anything to solve this quiz!
    
    However, feel free to play with the network! Can you also
    build a network that solves the equation below?
    
    (x + y) + y
    """

from miniflow import *

def Linear_Example():
    ## Simple Linear
#    inputs, weights, bias = Input(), Input(), Input()
#
#    f = Linear(inputs, weights, bias)
#
#    feed_dict = {
#        inputs: [6, 14, 3],
#        weights: [0.5, 0.25, 1.4],
#        bias: 2
#    }

    ## Linear
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}
    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    print(output) # should be 12.7 with this example


def Add_Example():
    ## Simple Addition
    #x, y = Input(), Input()
    #f = Add(x, y)
    #feed_dict = {x: 10, y: 5}

    ## Addition
    x, y, z = Input(), Input(), Input()
    f = Add(x, y, z)
    feed_dict = {x: 4, y: 5, z: 10}

    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(f, sorted_nodes)

    # NOTE: because topological_sort set the values for the `Input` nodes we could also access
    # the value for x with x.value (same goes for y).
    print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))


if __name__ == "__main__":
    Linear_Example()
#    Add_Example()
