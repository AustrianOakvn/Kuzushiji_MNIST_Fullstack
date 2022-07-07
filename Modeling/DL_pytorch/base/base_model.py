from turtle import forward
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Model prints with number of trainabble parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)