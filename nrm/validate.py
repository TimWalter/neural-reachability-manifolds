import argparse
from typing import Type

import torch
from tqdm import tqdm

from nrm.logger import Logger
from nrm.model import Model, Torus, OccupancyNetwork, MLP, Shell
from nrm.dataset.loader import ValidationSet


def main(model_class: Type[Model],
         model_id: int,
         batch_size: int):
    device = torch.device("cuda")

    validation_set = ValidationSet(batch_size, False)

    model = model_class.from_id(model_id).to(device)
    #model = torch.compile(model)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    logger = Logger(device, model_class, {}, 1, batch_size, -1, -1, None, validation_set,
                    validation_set, model, loss_function)

    model.eval()
    loss = 0.0
    for batch_idx, (morph, pose, label) in enumerate(tqdm(validation_set, desc=f"Validation")):
        morph = morph.to(device, non_blocking=True)
        pose = pose.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logit = model.predict(morph, pose)
        loss += loss_function(logit, label.float())
        logger.log_validation(0, batch_idx, morph, pose, label, logit, loss)
    loss /= len(validation_set) * batch_size
    logger.run.log(data=logger.assign_space(logger.compute_boundary_metrics(), "Boundaries"), step=logger.step+1, commit=True)

if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--model_id", type=int, default=1305)
    parser.add_argument("--batch_size", type=int, default=1000)

    args = parser.parse_args()
    args.model_class = eval(args.model_class)

    main(**vars(args))
