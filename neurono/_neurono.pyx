# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

cimport numpy as cnp
import numpy as np
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INT_t
ctypedef cnp.ulong_t INDEX_t
ctypedef cnp.uint8_t BOOL_t
from cpython cimport bool

FLOAT = np.float64
INT = np.intp
INDEX = np.uint
BOOL = np.uint8

from libc.math cimport exp

cdef class NeuralNetwork:
    cdef cnp.ndarray _nodes
    cdef cnp.ndarray _connections
    cdef cnp.ndarray _weights
    cdef cnp.ndarray _layers
    cdef cnp.ndarray _conn_layers
    cdef FLOAT_t rate
    cdef FLOAT_t thresh
    
    def __init__(NeuralNetwork self, layers, rate=0.1, thresh=0.000000001):
        self.rate = rate
        self.thresh = thresh
        n_layers = len(layers)
        self._layers = np.empty(shape=n_layers, dtype=INDEX)
        self._conn_layers = np.empty(shape=n_layers-1, dtype=INDEX)
        for i in range(n_layers):
            self._layers[i] = layers[i]
        n_nodes = 0
        for i in range(n_layers):
            n_nodes += self._layers[i]
        self._nodes = np.zeros(shape=(n_nodes,3),dtype=FLOAT)
        cdef INDEX_t n_connections = 0
        prev = self._layers[0]
        for i in range(1,n_layers):
            n_connections += prev * self._layers[i]
            self._conn_layers[i-1] = prev * self._layers[i]
            prev = self._layers[i]
            
        self._connections = np.empty(shape=(n_connections,2),dtype=INDEX)
        self._weights = np.empty(shape=n_connections,dtype=FLOAT)
        self.connect()
        
    cdef connect(NeuralNetwork self):
        '''Connect the layers of the network.'''
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INDEX_t n_layers = _layers.shape[0]
        cdef INDEX_t i
        cdef INDEX_t j
        cdef INDEX_t k
        cdef INDEX_t conn
        cdef INDEX_t layer_start
        cdef INDEX_t next_layer_start
        cdef INDEX_t from_ind
        cdef INDEX_t to_ind
        from_ind = 0
        to_ind = _layers[0]
        conn = 0
        layer_start = 0
        next_layer_start = 0
        for i in range(n_layers-1):
            next_layer_start += _layers[i]
            for j in range(_layers[i]):
                to_ind = next_layer_start
                for k in range(_layers[i+1]):
                    _connections[conn,0] = from_ind
                    _connections[conn,1] = to_ind
                    _weights[conn] = 1.0
                    conn += 1
                    to_ind += 1
                from_ind += 1
            layer_start += _layers[i]
    
    cdef input(NeuralNetwork self, cnp.ndarray[FLOAT_t, ndim=2] X, INDEX_t row):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INDEX_t n = X.shape[1]
        cdef INDEX_t i
        for i in range(_layers[0]):
            _nodes[i,1] = X[row,i]
        
    
    cdef prop(NeuralNetwork self):
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef cnp.ndarray[INDEX_t, ndim=1] _conn_layers = <cnp.ndarray[INDEX_t, ndim=1]> self._conn_layers
        
        cdef FLOAT_t weight
        cdef INDEX_t from_ind
        cdef INDEX_t to_ind
        cdef INDEX_t n_connections = _connections.shape[0]
        cdef INDEX_t i
        cdef INDEX_t j
        cdef INDEX_t n_layers = _layers.shape[0]
        cdef INDEX_t n_conn_layers = _conn_layers.shape[0]
        cdef INDEX_t conn = 0
        cdef INDEX_t layer_size
        cdef INDEX_t layer_to_update_start = _layers[0]
        for i in range(n_conn_layers):
            if i >= 1:
                for j in range(layer_to_update_start,layer_to_update_start+_layers[i]):
                    _nodes[j,1] = self.f(_nodes[j,0])
                layer_to_update_start += _layers[i]
            layer_size = _conn_layers[i]
            for j in range(layer_size):
                from_ind = _connections[conn,0]
                to_ind = _connections[conn,1]
                weight = _weights[conn]
                _nodes[to_ind,0] += weight*_nodes[from_ind,1]
                conn += 1
        
        for j in range(layer_to_update_start,layer_to_update_start+_layers[n_layers-1]):
            _nodes[j,1] = self.f(_nodes[j,0])
        
        
    cdef reset_nodes(NeuralNetwork self):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t i
        for i in range(n_nodes):
            _nodes[i,0] = 0.0
            _nodes[i,1] = 0.0
            _nodes[i,2] = 0.0
            
    cpdef cnp.ndarray[FLOAT_t, ndim=2] predict(NeuralNetwork self, cnp.ndarray[FLOAT_t, ndim=2] X):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef INDEX_t m = X.shape[0]
        cdef INDEX_t n = X.shape[1]
        cdef INDEX_t n_layers = self._layers.shape[0]
        cdef INDEX_t p = self._layers[n_layers-1]
        cdef INDEX_t row
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t output_start = n_nodes - p
        cdef INDEX_t j
        cdef INDEX_t k
        cdef cnp.ndarray[FLOAT_t, ndim=2] result = <cnp.ndarray[FLOAT_t, ndim=2]> np.empty(shape=(m,p),dtype=FLOAT)
        for row in range(m):
            self.reset_nodes()
            self.input(X, row)
            self.prop()
            k = 0
            for j in range(output_start, n_nodes):
                result[row,k] = _nodes[j,1]
                k += 1
        return result
    
    cdef FLOAT_t present(NeuralNetwork self, cnp.ndarray[FLOAT_t, ndim=2] y, INDEX_t row):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INDEX_t n = y.shape[1]
        cdef INDEX_t i
        cdef INDEX_t n_layers = _layers.shape[0]
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t output_size = _layers[n_layers - 1]
        cdef INDEX_t output_start = n_nodes - output_size
        cdef INDEX_t node
        cdef FLOAT_t pred
        cdef FLOAT_t error = 0.0
        for i in range(output_size):
            node = i + output_start
            pred = _nodes[node,1]
            _nodes[node,2] = (y[row,i] - pred)*self.f_prime(pred)
            error += 0.5*(y[row,i] - pred)**2
        return error
    
    cdef FLOAT_t f(NeuralNetwork self, FLOAT_t z):
        return 1.0 / (1.0 + exp(-1.0*z))
    
    cdef FLOAT_t f_prime(NeuralNetwork self, FLOAT_t y):
        return y*(1.0 - y)
    
    cdef back_prop(NeuralNetwork self):
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef cnp.ndarray[INDEX_t, ndim=1] _conn_layers = <cnp.ndarray[INDEX_t, ndim=1]> self._conn_layers
        
        cdef FLOAT_t weight
        cdef INDEX_t from_ind
        cdef INDEX_t to_ind
        cdef INDEX_t n_connections = _connections.shape[0]
        cdef INDEX_t i
        cdef INDEX_t j
        cdef INDEX_t n_layers = _layers.shape[0]
        cdef INDEX_t n_conn_layers = _conn_layers.shape[0]
        cdef INDEX_t conn = 0
        cdef INDEX_t layer_size
        cdef INDEX_t layer_to_update_start = n_connections - 1
        for i in range(n_conn_layers-1):
            for j in range(_conn_layers[i]):
                conn = layer_to_update_start - j
                from_ind = _connections[conn,0]
                to_ind = _connections[conn,1]
                _nodes[from_ind,2] = self.f_prime(_nodes[from_ind,1])*_nodes[to_ind,2]*_weights[conn]
            layer_to_update_start -= _conn_layers[i]
            
    cdef bool stop_check(NeuralNetwork self, FLOAT_t error):
        return error < self.thresh

cdef class BackpropNetwork(NeuralNetwork):
    cpdef fit(BackpropNetwork self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=2] y):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef INDEX_t m = X.shape[0]
        cdef INDEX_t n = X.shape[1]
        cdef INDEX_t n_layers = self._layers.shape[0]
        cdef INDEX_t p = self._layers[n_layers-1]
        cdef INDEX_t row
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t output_start = n_nodes - p
        cdef INDEX_t conn
        cdef INDEX_t from_ind
        cdef INDEX_t to_ind
        cdef FLOAT_t rate = 2.0 * self.rate
        cdef INDEX_t n_connections = _connections.shape[0]
        cdef FLOAT_t error = 0.0
        cdef FLOAT_t prev_error = 0.0
        cdef rate_increase = 1.2
        cdef rate_decrease = 0.5
        cdef cnp.ndarray[FLOAT_t, ndim=1] delta = <cnp.ndarray[FLOAT_t, ndim=1]> np.zeros(shape=n_connections,dtype=FLOAT)
        
        while True:
            error = 0.0
            for row in range(m):
                self.reset_nodes()
                self.input(X, row)
                self.prop()
                error += self.present(y,row)
                self.back_prop()
                for conn in range(n_connections):
                    from_ind = _connections[conn,0]
                    to_ind = _connections[conn,1]
                    delta[conn] += _nodes[to_ind,2] * _nodes[from_ind,1]
            
            if self.stop_check(error):
                break
            for conn in range(n_connections):
                _weights[conn] += rate * delta[conn] / m
                delta[conn] = 0.0
            if error > prev_error:
                rate *= rate_decrease
            elif error <= prev_error:
                rate *= rate_increase
            prev_error = error
            
cdef class BirdNetwork(NeuralNetwork):
    cdef cnp.ndarray _accumulators
    def __init__(self, layers, rate=0.1, thresh=0.000000001):
        NeuralNetwork.__init__(self, layers, rate, thresh)
        self._accumulators = np.zeros(shape=self._layers[-1],dtype=FLOAT)
    
    cdef reset_accumulators(BirdNetwork self):
        cdef cnp.ndarray[FLOAT_t, ndim=1] _accumulators = <cnp.ndarray[FLOAT_t, ndim=1]> self._accumulators
        cdef INDEX_t i
        cdef INDEX_t n = _accumulators.shape[0]
        for i in range(n):
            _accumulators[i] = 0.0
        
    cdef accumulate(BirdNetwork self):
        cdef cnp.ndarray[FLOAT_t, ndim=1] _accumulators = <cnp.ndarray[FLOAT_t, ndim=1]> self._accumulators
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t n_layers = _layers.shape[0]
        cdef INDEX_t n_outputs = _layers[n_layers - 1]
        cdef INDEX_t i
        for i in range(n_outputs):
            _accumulators[i] += _nodes[n_nodes - n_outputs + i]
    
    cdef FLOAT_t present(BirdNetwork self, cnp.ndarray[FLOAT_t, ndim=2] y, INDEX_t row):
        cdef cnp.ndarray[FLOAT_t, ndim=1] _accumulators = <cnp.ndarray[FLOAT_t, ndim=1]> self._accumulators
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INDEX_t n = y.shape[1]
        cdef INDEX_t i
        cdef INDEX_t n_layers = _layers.shape[0]
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t output_size = _layers[n_layers - 1]
        cdef INDEX_t output_start = n_nodes - output_size
        cdef INDEX_t node
        cdef FLOAT_t pred
        cdef FLOAT_t error = 0.0
        cdef FLOAT_t output
        for i in range(output_size):
            node = i + output_start
            pred = _accumulators[i]
            output = _nodes[node,1]
            _nodes[node,2] = (y[row,i] - pred)*self.f_prime(output)
            error += 0.5*(y[row,i] - pred)**2
        return error
    
    cpdef fit(BirdNetwork self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=2] y, cnp.ndarray[FLOAT_t, ndim=1] t, cnp.ndarray[INDEX_t, ndim=1] ids):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef INDEX_t m = X.shape[0]
        cdef INDEX_t n = X.shape[1]
        cdef INDEX_t n_layers = self._layers.shape[0]
        cdef INDEX_t p = self._layers[n_layers-1]
        cdef INDEX_t row
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t output_start = n_nodes - p
        cdef INDEX_t conn
        cdef INDEX_t from_ind
        cdef INDEX_t to_ind
        cdef FLOAT_t rate = 2.0 * self.rate
        cdef INDEX_t n_connections = _connections.shape[0]
        cdef FLOAT_t error = 0.0
        cdef FLOAT_t prev_error = 0.0
        cdef rate_increase = 1.2
        cdef rate_decrease = 0.5
        cdef cnp.ndarray[FLOAT_t, ndim=1] delta = <cnp.ndarray[FLOAT_t, ndim=1]> np.zeros(shape=n_connections,dtype=FLOAT)
        cdef INDEX_t current_id
        cdef INDEX_t first_row
        
        
        
        while True:
            row = 0
            y_row = 0
            while row < m:
                self.reset_accumulators()
                current_id = ids[row]
                first_row = row
                while ids[row] == current_id:
                    self.reset_nodes()
                    self.input(X, row)
                    self.prop()
                    self.accumulate()
                    row += 1
                error += self.present(y,y_row)
                row = first_row
                while ids[row] == current_id:
                    self.reset_nodes()
                    self.input(X, row)
                    self.prop()
                    self.back_prop()
                    row += 1
                    for conn in range(n_connections):
                        from_ind = _connections[conn,0]
                        to_ind = _connections[conn,1]
                        weight = _weights[conn]
                        delta[conn] += _nodes[to_ind,2] * _nodes[from_ind,1]
                y_row += 1
                
            if self.stop_check(error):
                break
            for conn in range(n_connections):
                _weights[conn] += rate * delta[conn] / m
                delta[conn] = 0.0
            if error > prev_error:
                rate *= rate_decrease
            elif error <= prev_error:
                rate *= rate_increase
            prev_error = error
            
