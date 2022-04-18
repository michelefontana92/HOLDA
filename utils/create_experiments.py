from xml.etree import ElementTree
import os
import json


def create_server_meta_tag(parent_node, node_id, base_save_path, is_proxy=False):
    idx = 0
    prefix = 'Proxy'if is_proxy else 'Server'
    meta_tag = ElementTree.Element('meta')

    id_tag = ElementTree.Element('id')
    id_tag.text = f'{prefix}{node_id}'
    meta_tag.insert(idx, id_tag)
    idx += 1

    log_tag = ElementTree.Element('log_path')
    log_tag.text = f'{base_save_path}/logs/log_{prefix}{node_id}.txt'
    meta_tag.insert(idx, log_tag)
    idx += 1

    best_ckpt_tag = ElementTree.Element('ckpt_best')
    best_ckpt_tag.text = f'{base_save_path}/{prefix}{node_id}Ckpt/{prefix}{node_id}_best.pt'
    meta_tag.insert(idx, best_ckpt_tag)
    idx += 1

    epoch_ckpt_tag = ElementTree.Element('ckpt_epoch')
    epoch_ckpt_tag.text = f'{base_save_path}/{prefix}{node_id}Ckpt/{prefix}{node_id}_epoch.pt'
    meta_tag.insert(idx, epoch_ckpt_tag)

    parent_node.insert(0, meta_tag)


def create_client_meta_tag(parent_node, node_id, base_save_path, data_path, data_names):

    idx = 0

    meta_tag = ElementTree.Element('meta')

    id_tag = ElementTree.Element('id')
    id_tag.text = f'Client_C{node_id}'
    meta_tag.insert(idx, id_tag)

    idx += 1

    log_tag = ElementTree.Element('log_path')
    log_tag.text = f'{base_save_path}/logs/log_clientC{node_id}.txt'
    meta_tag.insert(idx, log_tag)
    idx += 1

    best_ckpt_tag = ElementTree.Element('ckpt_best')
    best_ckpt_tag.text = f'{base_save_path}/ClientC{node_id}Ckpt/clientC{node_id}_best.pt'
    meta_tag.insert(idx, best_ckpt_tag)
    idx += 1

    epoch_ckpt_tag = ElementTree.Element('ckpt_epoch')
    epoch_ckpt_tag.text = f'{base_save_path}/ClientC{node_id}Ckpt/clientC{node_id}_epoch.pt'
    meta_tag.insert(idx, epoch_ckpt_tag)
    idx += 1

    train_path_tag = ElementTree.Element('train_path')
    train_path = data_names['train_path']
    train_path_tag.text = f'{data_path}/{train_path}'
    meta_tag.insert(idx, train_path_tag)
    idx += 1

    val_path_tag = ElementTree.Element('val_path')
    val_path = data_names['val_path']
    val_path_tag.text = f'{data_path}/{val_path}'
    meta_tag.insert(idx, val_path_tag)
    idx += 1

    test_path_tag = ElementTree.Element('test_path')
    test_path = data_names['test_path']
    test_path_tag.text = f'{data_path}/{test_path}'
    meta_tag.insert(idx, test_path_tag)
    idx += 1

    parent_node.insert(0, meta_tag)


def create_metadata_json(root, save_dir):
    metadata_dict = {}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    clients = root.find('architecture').find(
        'server').findall('client')

    for client_node in clients:
        id_tag = client_node.find('meta').find('id')
        best_ckpt_tag = client_node.find('meta').find('ckpt_best')
        train_path_tag = client_node.find('meta').find('train_path')
        val_path_tag = client_node.find('meta').find('val_path')
        test_path_tag = client_node.find('meta').find('test_path')

        metadata_dict[id_tag.text] = {}
        metadata_dict[id_tag.text]['models_dir'] = os.path.dirname(
            best_ckpt_tag.text)
        metadata_dict[id_tag.text]['train_path'] = train_path_tag.text
        metadata_dict[id_tag.text]['val_path'] = val_path_tag.text
        metadata_dict[id_tag.text]['test_path'] = test_path_tag.text

    with open(f'{save_dir}/metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=4, sort_keys=True)
