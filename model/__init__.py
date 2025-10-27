from typing import Union

import torch.nn
from .encoder import Encoder
from .decoder import Decoder


class Model(torch.nn.Module):

    def __init__(self, encoder_config: dict, decoder_config: dict,
                 output_func: Union[torch.nn.Sigmoid, torch.nn.ReLU] = None):
        super().__init__()
        # Does initialization with 0 make sense in this scenario? such that it predicts all unreachable initially
        self.output_func = output_func
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(encoding_dim=encoder_config["width"], **decoder_config)

    def forward(self, poses, mdh):
        mdh_encodings = self.encoder(mdh)
        output = self.decoder(poses, mdh_encodings)
        if self.output_func is None:
            return output
        else:
            return self.output_func(output)
