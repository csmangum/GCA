from models.cnn import AutomataCNN as AutomataCNN
from models.cnn import GeneralAutomataCNN as GeneralAutomataCNN
from models.evolution import ArtificialEvolution as ArtificialEvolution
from models.simple import SimpleSequentialNetwork as SimpleSequentialNetwork


class ModelFactory:
    """
    Factory class for creating models on the fly
    """

    def __init__(self, model_class: type) -> None:
        """
        Parameters
        ----------
        model_class : type
            The model class to instantiate
        """
        self.model_class = model_class

    def __call__(self) -> type:
        return self.model_class()
