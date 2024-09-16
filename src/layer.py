from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        # TODO: return the output of the layer
        pass

    def backward(self, *args, **kwargs):
        # TODO: return the gradient w.r.t outputs of previous layer and,
        #       if necessary, update weigths 
        pass