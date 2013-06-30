import numpy
from neurono import NeuralNetwork
from nose.tools import assert_true, assert_almost_equal, assert_not_equal, assert_equal



class TestNeuralNetwork(object):
    
    def __init__(self):
        pass
    
    def test_predict(self):
        X = numpy.array([[2.0]])
        y = numpy.array(1.0 / (1.0 + numpy.exp(-1.0/(1.0+numpy.exp(-X)))))
        net = NeuralNetwork([1,1,1])
        assert_almost_equal(net.predict(X)[0,0],y[0,0])
        
    def test_fit(self):
        X = numpy.array([[2.0]])
        y = numpy.array(1.0 / (1.0 + numpy.exp(-1.0/(1.0+numpy.exp(-X)))))
        thresh = .0000001
        net = NeuralNetwork([1,2,1],thresh=thresh)
        net.fit(X,y)
        error = 0.5*((net.predict(X)[0,0] - y[0,0])**2)
        assert_true(error < thresh)
    
    def test_fit_harder(self):
        X = numpy.random.normal(size=(100,10))
        beta = numpy.random.normal(size=10)
        y = (numpy.dot(X,beta)).reshape((100,1))
        y = 1.0 / (1.0 + numpy.exp(-y))
        print y
        net = NeuralNetwork([10,50,1])
        net.fit(X,y)
        pred = net.predict(X)
        error = numpy.mean(0.5*((X-pred)**2))
        
        
        
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])