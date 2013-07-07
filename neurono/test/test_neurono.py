import numpy
from neurono import BackpropNetwork
from nose.tools import assert_true, assert_almost_equal, assert_not_equal, assert_equal



class TestBackpropNetwork(object):
    
    def __init__(self):
        pass
    
    def test_predict(self):
        X = numpy.array([[2.0]])
        y = numpy.array(1.0 / (1.0 + numpy.exp(-1.0/(1.0+numpy.exp(-X)))))
        net = BackpropNetwork([1,1,1])
        assert_almost_equal(net.predict(X)[0,0],y[0,0])
        
    def test_fit(self):
        X = numpy.array([[2.0, 3.0]])
        y = numpy.array(1.0 / (1.0 + numpy.exp(-1.0/(1.0+numpy.exp(-numpy.dot(X,X.transpose()))))))
        thresh = .0000001
        net = BackpropNetwork([2,4,1],thresh=thresh)
        net.fit(X,y)
        error = 0.5*((net.predict(X)[0,0] - y[0,0])**2)
        assert_true(error < thresh)

class TestBirdNetwork(object):
    pass

        
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])