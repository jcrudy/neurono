from neurono import InputNode, SigmoidNode, FeedForwardNet
from nose.tools import assert_true, assert_almost_equal, assert_not_equal, assert_equal
from math import exp

class TestNode(object):
    def __init__(self):
        self.in_node = InputNode()
        self.out_node = SigmoidNode()
        self.in_node.connect(self.out_node)
        self.in_node.outputs[0].weight = 2.0
        
    def test_connect(self):
        assert_true(self.out_node in self.in_node.children())
        assert_true(self.in_node in self.out_node.parents())
        
    def test_update(self):
        self.in_node.present(2.0)
        self.in_node.update()
        assert_almost_equal(self.in_node.outputs[0].value, 4.0)
        self.out_node.update()
        assert_almost_equal(self.out_node.value, 1.0 / (1.0 + exp(-4.0)))
    
class TestNet(object):
    def __init__(self):
        self.net = FeedForwardNet(1,1,1,thresh=.000000000000001)
        
    def test_predict(self):
        X = [[2.0]]
        assert_equal(self.net.predict(X)[0], [1.0 / (1.0 + exp(-1*(1.0 / (1.0 + exp(-2.0)))))])
        
    def test_randomize(self):
        X = [[2.0]]
        v1 = self.net.predict(X)
        self.net.randomize()
        v2 = self.net.predict(X)
        self.net.randomize()
        v3 = self.net.predict(X)
        assert_not_equal(v1, v2)
        assert_not_equal(v2, v3)
        
    def test_fit(self):
        X = [[2.0],[10.0]]
        y = [[1/(1+exp(-3.0))],[1/(1+exp(-10.0))]]
        self.net.randomize()
        self.net.fit(X,y)
        
        pred = self.net.predict(X)
        for i in range(len(y)):
            assert_almost_equal(pred[i][0],y[i][0])
        
        
        
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])