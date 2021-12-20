import functools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from buildingBlocks.remote_client_nn import RemoteLocalClient_NN
from buildingBlocks.remote_tproxy import RemoteTrustedProxy
from buildingBlocks.server import Server
import ray
from metadata.meta import ClientConfig, HP_Training_Client, HP_Training_Server, Metadata_Client, Metadata_Server, ProxyConfig, ServerConfig
import time
import inspect
from importlib import import_module
import torch


def str2bool(string):
    if string.rstrip().lower() == 'false':
        return False
    else:
        return True


def str2function(string, params):
    def get_members(module):
        return [o for o in inspect.getmembers(module) if inspect.isfunction(o[1])]

    split = string.split('.')
    fn_name = split[-1]
    if len(split) > 1:
        module_path = '.'.join(split[:-1])
        module = import_module(module_path)
        return functools.partial(getattr(module, fn_name), **params)
    else:
        return functools.partial(locals()[fn_name],
                                 **params)


def parse_function_tag(tree, tag_fn_name):
    result_dict = {}
    try:
        for params in tree:
            if params.tag == tag_fn_name:
                model_fn = params.text.rstrip()
            else:
                result_dict = {}
                for param in params:
                    try:
                        result_dict[param.tag] = int(
                            param.text.rstrip('\n'))
                    except ValueError:
                        result_dict[param.tag] = float(
                            param.text.rstrip('\n'))
    except TypeError:
        raise KeyError(f'Tag <{tree.tag}> not found')
    return str2function(model_fn, result_dict)


def parse_task_tag(tree):
    task_tag = tree.find('task')
    result_dict = {}
    try:
        for task in task_tag:
            try:
                result_dict[task.tag] = int(task.text)
            except ValueError:
                result_dict[task.tag] = task.text
    except TypeError:
        raise KeyError('Tag <task> not found')
    return result_dict


def parse_metrics_tag(tree):
    metrics_tag = tree.find('metrics')
    result_dict = {}
    n_classes = parse_task_tag(tree)['n_classes']
    try:
        for metric in metrics_tag:
            text = metric.text.rstrip().lower()
            if text == 'accuracy':
                result_dict['accuracy'] = accuracy_score
            elif text == 'precision':
                result_dict['precision'] = functools.partial(precision_score,
                                                             labels=list(
                                                                 range(n_classes)),
                                                             average='weighted',
                                                             zero_division=0)

            elif text == 'recall':
                result_dict['recall'] = functools.partial(recall_score,
                                                          labels=list(
                                                              range(n_classes)),
                                                          average='weighted',
                                                          zero_division=0)
            elif text == 'f1':
                result_dict['f1'] = functools.partial(f1_score,
                                                      labels=list(
                                                          range(n_classes)),
                                                      average='weighted',
                                                      zero_division=0)

    except TypeError:
        result_dict['accuracy'] = accuracy_score
        result_dict['precision'] = functools.partial(precision_score,
                                                     labels=list(
                                                         range(n_classes)),
                                                     average='weighted',
                                                     zero_division=0)
        result_dict['recall'] = functools.partial(recall_score,
                                                  labels=list(
                                                      range(n_classes)),
                                                  average='weighted',
                                                  zero_division=0)
        result_dict['f1'] = functools.partial(f1_score,
                                              labels=list(range(n_classes)),
                                              average='weighted',
                                              zero_division=0)
    return result_dict


def parse_setting_tag(tree):
    setting_tag = tree.find('setting')
    result_dict = {}
    keys = ['use_deltas', 'use_state']
    try:
        for setting in setting_tag:
            result_dict[setting.tag] = str2bool(setting.text)

    except TypeError:
        result_dict['use_deltas'] = True
        result_dict['use_state'] = True

    for k in keys:
        if k not in result_dict.keys():
            result_dict[k] = True
    return result_dict


def _parse_children(tree, tag):
    result_dict = {}
    function_tags = {
        'model': 'model_fn',
        'optimizer': 'optimizer_fn',
        'loss': 'loss_fn',
        'aggregation': 'aggregation_fn',
        'init_model': 'init_model_fn',
        'extract_model': 'extract_model_fn'
    }

    try:
        meta_tag = tree.find(tag)
        for meta in meta_tag:
            if meta.tag in function_tags.keys():
                result_dict[function_tags[str(meta.tag)]] = parse_function_tag(
                    meta, function_tags[str(meta.tag)])
            else:
                result_dict[meta.tag] = meta.text
    except TypeError:
        return {}
    return result_dict


def parse_model_tag(tree):
    result_dict = {}
    meta_tag = tree.find('model')
    result_dict = parse_function_tag(meta_tag, 'model_fn')
    return result_dict


def parse_server_tag(tree, parent):
    result_dict = {}
    for child in tree:
        meta_dict = _parse_children(tree, child.tag)
        result_dict[child.tag] = {}
        for key, value in meta_dict.items():
            result_dict[child.tag][key] = value

    training = result_dict['training']
    convert_to_int = ['epochs', 'patience', 'epoch2ckpt']
    convert_to_float = ['sample_size']
    convert_to_bool = ['from_check']
    convert_to_function = ['optimizer_fn', 'loss_fn']

    for key, value in training.items():
        if key in convert_to_int:
            training[key] = int(value)
        elif key in convert_to_float:
            training[key] = float(value)
        elif key in convert_to_bool:
            training[key] = str2bool(training[key])
        elif key in convert_to_function:
            training[key] = str2function()
    result_dict['training']['build_model_fn'] = parse_function_tag(
        parent.find('model'), 'model_fn')
    result_dict['training'] = training

    use_deltas = parse_setting_tag(parent)['use_deltas']
    target_label = parse_task_tag(parent)['target']
    metadata = result_dict['meta']
    training = result_dict['training']
    server_meta = Metadata_Server(**metadata)
    server_training = HP_Training_Server(**training)
    config_dict = {
        'metadata': server_meta,
        'training_params': server_training,
        'use_deltas': use_deltas,
        'target_label': target_label
    }
    config = ServerConfig(**config_dict)
    print()
    print(50*'-')
    print('Server Config: ', config)
    print(50*'-')
    print()
    children_list = []
    num_children = 0
    for child in tree:
        if child.tag == 'proxy':
            proxy, n_proxy_children = parse_proxy_tag(child, parent)
            num_children += n_proxy_children + 1
            children_list.append(proxy)

        elif child.tag == 'client':
            client = parse_client_tag(child, parent)
            num_children += 1
            children_list.append(client)

    print('Num Children = ', num_children)
    ray.init(num_cpus=num_children)
    server = Server(config, children_list)
    return server


def parse_proxy_tag(tree, parent):
    result_dict = {}
    for child in tree:
        meta_dict = _parse_children(tree, child.tag)
        result_dict[child.tag] = {}
        for key, value in meta_dict.items():
            result_dict[child.tag][key] = value

    training = result_dict['training']
    convert_to_int = ['epochs', 'patience', 'epoch2ckpt']
    convert_to_float = ['sample_size']
    convert_to_bool = ['from_check']

    for key, value in training.items():
        if key in convert_to_int:
            training[key] = int(value)
        elif key in convert_to_float:
            training[key] = float(value)
        elif key in convert_to_bool:
            training[key] = str2bool(training[key])
    result_dict['training']['build_model_fn'] = parse_function_tag(
        parent.find('model'), 'model_fn')
    result_dict['training'] = training

    use_deltas = parse_setting_tag(parent)['use_deltas']
    use_state = parse_setting_tag(parent)['use_state']
    target_label = parse_task_tag(parent)['target']
    metadata = result_dict['meta']
    training = result_dict['training']
    server_meta = Metadata_Server(**metadata)
    server_training = HP_Training_Server(**training)
    config_dict = {
        'metadata': server_meta,
        'training_params': server_training,
        'use_deltas': use_deltas,
        'target_label': target_label,
        'use_state': use_state
    }

    config = ProxyConfig(**config_dict)
    print()
    print(50*'-')
    print('Proxy Config: ', config)
    print(50*'-')
    print()
    children_list = []
    num_children = 0
    for child in tree:
        if child.tag == 'proxy':
            proxy, n_proxy_children = parse_proxy_tag(child, parent)
            num_children += n_proxy_children + 1
            children_list.append(proxy)
        elif child.tag == 'client':
            client = parse_client_tag(child, parent)
            num_children += 1
            children_list.append(client)
    server = functools.partial(
        RemoteTrustedProxy.remote, config, children_list)
    return server, num_children


def parse_client_tag(tree, parent):
    result_dict = {}
    for child in tree:
        meta_dict = _parse_children(tree, child.tag)
        result_dict[child.tag] = {}
        for key, value in meta_dict.items():
            result_dict[child.tag][key] = value

    training = result_dict['training']
    convert_to_int = ['epochs', 'patience', 'epoch2ckpt', 'batch_size']
    convert_to_float = []
    convert_to_bool = ['use_weights']
    for key, value in training.items():
        if key in convert_to_int:
            training[key] = int(value)
        elif key in convert_to_float:
            training[key] = float(value)
        elif key in convert_to_bool:
            training[key] = str2bool(training[key])

    result_dict['training']['build_model_fn'] = parse_function_tag(
        parent.find('model'), 'model_fn')
    result_dict['training'] = training
    result_dict['meta']['metrics'] = parse_metrics_tag(parent)
    result_dict['meta']['n_classes'] = parse_task_tag(parent)['n_classes']
    metadata = result_dict['meta']
    training = result_dict['training']
    server_meta = Metadata_Client(**metadata)
    server_training = HP_Training_Client(**training)
    config_dict = {
        'metadata': server_meta,
        'training_params': server_training
    }

    config = ClientConfig(**config_dict)
    print()
    print(50*'-')
    print('Client Config: ', config)
    print(50*'-')
    print()

    server = functools.partial(
        RemoteLocalClient_NN.remote, config)
    return server


def parse_architecture_tag(tree):

    try:
        arch_tag = tree.find('architecture')
        server_tree = arch_tag.find('server')
        server = parse_server_tag(server_tree, tree)

    except AttributeError:
        raise KeyError('Tag <architecture> not found')
    except KeyError as e:
        raise KeyError(e)
    return server
