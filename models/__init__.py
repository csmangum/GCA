from models.cnn import AutomataCNN as AutomataCNN
from models.cnn import GeneralAutomataCNN as GeneralAutomataCNN
from models.evolution import ArtificialEvolution as ArtificialEvolution
from models.simple import SimpleSequentialNetwork as SimpleSequentialNetwork
from models.transformer import SimpleTransformer as SimpleTransformer


class ModelFactory:
    """
    Factory class for creating models on the fly
    """

    def __init__(self, model_class: type, arguments: dict) -> None:
        """
        Parameters
        ----------
        model_class : type
            The model class to instantiate
        arguments : dict
            The arguments to pass to the model class
        """
        self.model_class = model_class
        self.arguments = arguments

    def __call__(self) -> type:
        return self.model_class(**self.arguments)
