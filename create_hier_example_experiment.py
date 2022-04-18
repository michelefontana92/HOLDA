from xml.etree import ElementTree
import os
from utils.create_experiments import create_client_meta_tag, create_metadata_json, create_server_meta_tag


if __name__ == '__main__':

    base_path = 'architectures/baselines/example_hier_base.xml'

    base_save_path = './examples/experiments/hier'
    data_names = {
        'train_path': 'adult_train_enc.csv',
        'val_path': 'adult_val_enc.csv',
        'test_path': 'adult_test_enc.csv',
    }
    tree = ElementTree.parse(base_path)
    root = tree.getroot()
    server = root.find('architecture').find(
        'server')
    create_server_meta_tag(
        server, 'S', base_save_path)

    proxies = root.find('architecture').find(
        'server').findall('proxy')

    node_id = 0
    for pid, proxy in enumerate(proxies):
        create_server_meta_tag(
            proxy, f'P{pid}', base_save_path, is_proxy=True)

        clients = proxy.findall('client')

        for client in clients:
            data_path = f'./examples/datasets/node_{node_id}'
            create_client_meta_tag(
                client, node_id, base_save_path,
                data_path, data_names)
            node_id += 1

    save_dir = './architectures/holda'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tree.write(
        f'{save_dir}/holda_hier.xml')

    create_metadata_json(
        root, f'{base_save_path}')
