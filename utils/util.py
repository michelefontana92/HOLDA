import os


def create_model_name(path, global_id, local_id):
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    filename_split = filename.split('.')
    save_model_path = f'{base_dir}/{filename_split[0]}_{global_id}_{local_id}.{filename_split[1]}'
    return save_model_path


def create_model_name_state(path, state_id):
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    filename_split = filename.split('.')
    save_model_path = f'{base_dir}/state_{filename_split[0]}_{state_id}.{filename_split[1]}'
    return save_model_path
