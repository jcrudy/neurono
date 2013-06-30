import random
from math import exp
from itertools import chain

class Net(object):
    pass

class Node(object):
    def __init__(self):
        self.activation = 0.0
        self.bias = 0.0
        self.outputs = []
        self.inputs = []
        self.sigma = 0.0
        self.value = self.f()
        self.back_error = 0.0
        self.error = 0.0
    
    def connect(self, other):
        conn = Connection(self, other)
        self.outputs.append(conn)
        other._reverse_connect(conn)
        
    def _reverse_connect(self, connection):
        self.inputs.append(connection)

    def f(self):
        raise NotImplementedError
    
    def f_prime(self):
        raise NotImplementedError
    
    def update(self):
        self.sigma = 0.0
        for conn in self.inputs:
            self.sigma += conn.value
        self.value = self.f()
        for conn in self.outputs:
            conn.update()
    
    def present(self, example):
        self.error = self.value - example
        self.back_error = self.f_prime() * self.error
        
    def back_prop(self):
        self.back_error = 0.0
        for conn in self.outputs:
            self.back_error += conn.target.back_error
        self.back_error *= self.f_prime()
    
    def parents(self):
        for conn in self.inputs:
            yield conn.source
    
    def children(self):
        for conn in self.outputs:
            yield conn.target
            
    def randomize(self):
        for conn in self.outputs:
            conn.randomize()
        
class SigmoidNode(Node):
    def f(self):
        return 1.0 / (1.0 + exp(-self.sigma))
    
    def f_prime(self):
        return self.value*(1.0 - self.value)

class InputNode(Node):
    def __init__(self):
        self.outputs = []
        self.value = 0.0
    
    def present(self, value):
        self.value = value
    
    def update(self):
        for conn in self.outputs:
            conn.update()

class Connection(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.value = 0.0
        self.weight = 1.0
    
    def randomize(self):
        self.weight = random.gauss(0.0,1.0)
        
    def connect(self, source, target):
        self.source = source
        self.target = target
        
    def update(self):
        self.value = self.source.value*self.weight

    def delta(self):
        return self.target.back_error * self.source.value
        
class Layer(object):
    def __init__(self):
        self.nodes = []
    
    def __iter__(self):
        return iter(self.nodes)
    
    def connect(self, other):
        for node in self:
            for other_node in other:
                node.connect(other_node)
    
    def update(self):
        for node in self:
            node.update()
    
    def present(self, data):
        for i, val in enumerate(data):
            self.nodes[i].present(val)
    
    def append(self, node):
        self.nodes.append(node)
    
    def values(self):
        return [node.value for node in self]
    
    def randomize(self):
        for node in self:
            node.randomize()
    
    @property
    def connections(self):
        for node in self:
            for conn in node.outputs:
                yield conn
                
    def __len__(self):
        return len(self.nodes)
    
    def back_prop(self):
        for node in self:
            node.back_prop()
    
class Trainer(object):
    
    def train(self, net, data):
        raise NotImplementedError
    

class FeedForwardNet(Net):
    def __init__(self, inputs, hidden, outputs, thresh=.01, rate=.01):
        self.inputs = Layer()
        self.hidden = Layer()
        self.outputs = Layer()
        for _ in range(inputs):
            self.inputs.append(InputNode())
        for _ in range(hidden):
            self.hidden.append(SigmoidNode())
        for _ in range(outputs):
            self.outputs.append(SigmoidNode())
        self.inputs.connect(self.hidden)
        self.hidden.connect(self.outputs)
        self.n = len(self.inputs)
        self.p = len(self.outputs)
        self.thresh = thresh
        self.rate = rate
        self.increase = False
        self.last_err = None
        
    @property
    def error(self):
        err = 0.0
        for conn in self.outputs:
            err += conn.error**2
        return err
    
    def stop_check(self):
        err = self.error
        if self.last_err is not None and err > self.last_err:
            self.increase = True
        self.last_err = err
        print 'rate:', self.rate
        print self.last_err
        return self.last_err < self.thresh
    
    def delta(self):
        result = []
        for conn in self.connections:
            result.append(conn.delta())
#        for i in range(m):
#            result.append([])
#            self.outputs.present(y[i])
#            self.hidden.back_prop()
#            for conn in self.connections:
#                result[i].append(conn.delta())
        return result
    
    @property
    def connections(self):
        for conn in self.inputs.connections:
            yield conn
        for conn in self.hidden.connections:
            yield conn
    
    @property
    def weights(self):
        for conn in self.inputs.connections:
            yield conn.weight
        for conn in self.hidden.connections:
            yield conn.weight

    @weights.setter
    def weights(self, values):
        for i, conn in enumerate(chain(self.inputs, self.hidden)):
            conn.weight = values[i]
    
    def fit(self, X, y):
        m = len(X)
        assert m == len(y)
        first = True
        while first or not self.stop_check():
            for i in range(m):
                self.prop(X[i])
                self.outputs.present(y[i])
                if first:
                    delta = self.delta()
                    first = False
                else:
                    delta_ = self.delta()
                    for j in range(len(delta)):
                        delta[j] += delta_[j]
            for j, conn in enumerate(self.connections):
                conn.weight += -self.rate*delta[j]
            if self.increase:
                self.rate *= .5
                print 'hi'
            else:
                self.rate *= 1.1
    
    def prop(self, data):
        self.inputs.present(data)
        for node in self.nodes():
            node.update()
        return self.outputs.values()
    
    def predict(self, X):
        m = len(X)
        result = []
        for i in range(m):
            result.append(self.prop(X[i]))
        return result
    
    def randomize(self):
        for node in self.nodes():
            node.randomize()
    
    def nodes(self):
        for node in self.inputs:
            yield node
        for node in self.hidden:
            yield node
        for node in self.outputs:
            yield node
            
#    
#
#class BackPropagationTrainer(Trainer):
#    def __init__(self, net, X, y):
#        self.net = net
#        self.X = X
#        self.y = y
#        
#    def iterate(self):
#        pred = self.net.predict(self.X)
#        diff = self.y - pred
#        
#    
#    def train(self):
#        
#        
#        
        
        
        
        
        
        
        
        
        