import attr
from aggregations.nn_aggregation import aggregate_nn_weighted, extract_model_nn, init_model_nn
import functools
import torch.optim as optim
import torch.nn as nn
from models.nn import create_net


@attr.s(eq=False, frozen=False, slots=True)
class HP_Training_Server:
    build_model_fn = attr.ib(default=functools.partial(create_net),
                             validator=[attr.validators.instance_of(functools.partial)])
    epochs = attr.ib(default=1,
                     validator=[attr.validators.instance_of(int)])
    patience = attr.ib(default=10,
                       validator=[attr.validators.instance_of(int)])
    sample_size = attr.ib(default=1.0,
                          validator=[attr.validators.instance_of(float)])

    aggregation_fn = attr.ib(default=functools.partial(aggregate_nn_weighted),
                             validator=[attr.validators.instance_of(functools.partial)])

    init_model_fn = attr.ib(default=functools.partial(init_model_nn),
                            validator=[attr.validators.instance_of(functools.partial)])
    extract_model_fn = attr.ib(default=functools.partial(extract_model_nn),
                               validator=[attr.validators.instance_of(functools.partial)])

    epoch2ckpt = attr.ib(default=10,
                         validator=[attr.validators.instance_of(int)])
    from_check = attr.ib(default=False,
                         validator=[attr.validators.instance_of(bool)])
    starting_model_path = attr.ib(default='',
                                  validator=[attr.validators.instance_of(str)])


@attr.s(eq=False, frozen=False, slots=True)
class Metadata_Server:
    id = attr.ib(default='Default_Server.txt',
                 validator=[attr.validators.instance_of(str)])
    log_path = attr.ib(default='default_log.txt',
                       validator=[attr.validators.instance_of(str)])
    ckpt_best = attr.ib(default='default_ckpt_best.pt',
                        validator=[attr.validators.instance_of(str)])
    ckpt_epoch = attr.ib(default='default_ckpt_epoch.pt',
                         validator=[
                             attr.validators.instance_of(str)])


@attr.s(eq=False, frozen=False, slots=True)
class HP_Training_Client:
    build_model_fn = attr.ib(default=functools.partial(create_net),
                             validator=[attr.validators.instance_of
                                        (functools.partial)])
    optimizer_fn = attr.ib(default=functools.partial(
        functools.partial(optim.Adam, lr=1e-4)),
        validator=[attr.validators.instance_of(functools.partial)])
    loss_fn = attr.ib(default=functools.partial(
        functools.partial(nn.CrossEntropyLoss)),
        validator=[attr.validators.instance_of(functools.partial)])
    epochs = attr.ib(default=1,
                     validator=[attr.validators.instance_of(int)])
    batch_size = attr.ib(default=128,
                         validator=[attr.validators.instance_of(int)])
    patience = attr.ib(default=10,
                       validator=[attr.validators.instance_of(int)])
    epoch2ckpt = attr.ib(default=10,
                         validator=[attr.validators.instance_of(int)])
    use_weights = attr.ib(default=True,
                          validator=[attr.validators.instance_of(bool)])


@attr.s(eq=False, frozen=False, slots=True)
class Metadata_Client:
    id = attr.ib(default='Default_Client.txt',
                 validator=[attr.validators.instance_of(str)])
    log_path = attr.ib(default='default_log.txt',
                       validator=[attr.validators.instance_of(str)])
    ckpt_best = attr.ib(default='default_ckpt_best.txt',
                        validator=[attr.validators.instance_of(str)])
    ckpt_epoch = attr.ib(default='default_ckpt_epoch.txt',
                         validator=[attr.validators.instance_of(str)])
    metrics = attr.ib(default={},
                      validator=[attr.validators.instance_of(dict)])
    n_classes = attr.ib(default=2,
                        validator=[attr.validators.instance_of(int)])

    train_path = attr.ib(default='',
                         validator=[attr.validators.instance_of(str)])

    val_path = attr.ib(default='',
                       validator=[attr.validators.instance_of(str)])
    test_path = attr.ib(default='',
                        validator=[attr.validators.instance_of(str)])


@ attr.s(eq=False, frozen=False, slots=True)
class CheckPoint:
    model = attr.ib()
    train_metrics = attr.ib()
    val_metrics = attr.ib()


@ attr.s(eq=False, frozen=False, slots=True)
class ServerConfig:

    metadata = attr.ib(default=Metadata_Server(),
                       validator=[attr.validators.instance_of(Metadata_Server)])
    training_params = attr.ib(default=HP_Training_Server(),
                              validator=[attr.validators.instance_of(HP_Training_Server)])
    use_deltas = attr.ib(default=True,
                         validator=[attr.validators.instance_of(bool)])
    target_label = attr.ib(default='class',
                           validator=[attr.validators.instance_of(str)])


@ attr.s(eq=False, frozen=False, slots=True)
class ProxyConfig(ServerConfig):
    pers_training_params = attr.ib(default=HP_Training_Server(),
                                   validator=[attr.validators.instance_of(HP_Training_Server)])


@ attr.s(eq=False, frozen=False, slots=True)
class ClientConfig:
    metadata = attr.ib(default=Metadata_Client(),
                       validator=[attr.validators.instance_of(Metadata_Client)])

    training_params = attr.ib(default=HP_Training_Client(),
                              validator=[attr.validators.instance_of(HP_Training_Client)])
    pers_training_params = attr.ib(default=HP_Training_Client(),
                                   validator=[attr.validators.instance_of(HP_Training_Client)])
