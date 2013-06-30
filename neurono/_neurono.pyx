# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False


cdef class NeuralNetwork:
    cdef cnp.ndarray _nodes
    cdef cnp.ndarray _connections
    cdef cnp.ndarray _weights
    cdef cnp.ndarray _layers
    cdef cnp.ndarray _conn_layers
    
    def __init__(NeuralNetwork self, layers):
        n_layers = len(layers)
        self._layers = np.empty(shape=n_layers, dtype=INDEX_t)
        self._conn_layers = np.empty(shape=n_layers-1, dtype=INDEX_t)
        for i in range(n_layers):
            self._layers[i] = layers[i]
        n_nodes = 0
        for i in range(n_layers):
            n_nodes += self._layers[i]
        self._nodes = np.zeros(shape=(n_nodes,3),dtype=FLOAT)
        n_connections = 0
        prev = 0
        for i in range(n_layers):
            n_connections += prev * i
        self._connections = np.empty(shape=(n_connections,2),INDEX)
        self._weights = np.empty(shape=n_connections,FLOAT)
        self.connect()
        
            
    cpdef connect(NeuralNetwork self):
        '''Connect the layers of the network.'''
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INT n_layers = _layers.shape[0]
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int k
        cdef unsigned int conn
        cdef unsigned int layer_start
        cdef unsigned int next_layer_start
        cdef unsigned int from_ind
        cdef unsigned int to_ind
        from_ind = 0
        to_ind = _layers[0]
        
        conn = 0
        layer_start = 0
        next_layer_start = 0
        for i in range(n_layers-1):
            next_layer_start += _layers[i]
            for j in range(_layers[i]):
                from_ind += 1
                to_ind = next_layer_start
                for k in range(_layers[i+1]):
                    conn += 1
                    to_ind += 1
                    _connections[conn,0] = from_ind
                    _connections[conn,1] = to_ind
                    weights[conn] = 0.1
            layer_start += _layers[i]
    
    cpdef input(NeuralNetwork self, cnp.ndarray[FLOAT, ndim=2] X, INDEX_t row):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef INDEX_t n = X.shape[1]
        cdef INDEX_t i
        for i in range(_layers[0]):
            _nodes[i,1] = X[row,i]
        
    
    cpdef prop(NerualNetwork self):
        cdef cnp.ndarray[INDEX_t, ndim=2] _connections = <cnp.ndarray[INDEX_t, ndim=2]> self._connections
        cdef cnp.ndarray[FLOAT_t, ndim=1] _weights = <cnp.ndarray[FLOAT_t, ndim=1]> self._weights
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef cnp.ndarray[INDEX_t, ndim=1] _layers = <cnp.ndarray[INDEX_t, ndim=1]> self._layers
        cdef cnp.ndarray[INDEX_t, ndim=1] _conn_layers = <cnp.ndarray[INDEX_t, ndim=1]> self._conn_layers
        
        cdef FLOAT_t weight
        cdef INDEX_t from_ind
        cdef INDEX_t to_ind
        cdef INDEX_t n_connections = _connections.shape[0]
        cdef INDEX_t conn
        cdef INDEX_t i
        cdef INDEX_t j
        cdef INDEX_t n_conn_layers = _conn_layers.shape[0]
        cdef INDEX_t conn = 0
        for i in range(n_conn_layers):
            
            if i >= 1
                for j in range(_layers[i]):
                    _nodes[j,1] = self.f(_nodes[j,0])
            
            layer_size = _conn_layers[i]
            for j in range(layer_size):
                from_ind = _connections[conn,0]
                to_ind = _connections[conn,1]
                weight = _weights[conn]
                _nodes[to_ind,1] += weight*_nodes[from_ind,0]
                conn += 1
    
    cpdef reset_nodes(NeuralNetwork self):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t i
        for i in range(n_nodes):
            _nodes[i,0] = 0.0
            _nodes[i,1] = 0.0
            _nodes[i,2] = 0.0
            
    cpdef predict(NeuralNetwork self, cnp.ndarray[FLOAT_t, ndim=2] X):
        cdef cnp.ndarray[FLOAT_t, ndim=2] _nodes = <cnp.ndarray[FLOAT_t, ndim=2]> self._nodes
        cdef INDEX_t m X.shape[0]
        cdef INDEX_t n X.shape[1]
        cdef INDEX_t p self.layers[len(self.layers)-1]
        cdef INDEX_t row
        cdef INDEX_t n_nodes = _nodes.shape[0]
        cdef INDEX_t output_start = n_nodes - p
        cdef cnp.ndarray[FLOAT_t, ndim=2] result = <cnp.ndarray[FLOAT_t, ndim=2]> np.empty(shape=(m,p),dtype=FLOAT)
        for row in range(m):
            self.reset_nodes()
            self.input(X, row)
            self.prop()
            for j in range(output_start, n_nodes):
                result[row,j] = _nodes[j,1]
        return result
    
        
    
class InputNode:
    pass

class OutputNode:
    pass

class Connection:
    pass
