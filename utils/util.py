import os


def create_model_name(path, global_id, local_id, personalized=False):
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    filename_split = filename.split('.')
    if personalized:
        save_model_path = f'{base_dir}/personalized_{filename_split[0]}_{global_id}_{local_id}.{filename_split[1]}'
    else:
        save_model_path = f'{base_dir}/{filename_split[0]}_{global_id}_{local_id}.{filename_split[1]}'

    return save_model_path


def create_model_name_state(path, state_id, personalized=False):
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    filename_split = filename.split('.')
    if personalized:
        save_model_path = f'{base_dir}/personalized_state_{filename_split[0]}_{state_id}.{filename_split[1]}'
    else:
        save_model_path = f'{base_dir}/state_{filename_split[0]}_{state_id}.{filename_split[1]}'
    return save_model_path


def create_model_name_monitor(path, state_id, input=True):
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    filename_split = filename.split('.')
    if input:
        save_model_path = f'{base_dir}/monitor_{filename_split[0]}_{state_id}_in.{filename_split[1]}'
    else:
        save_model_path = f'{base_dir}/monitor_{filename_split[0]}_{state_id}_out.{filename_split[1]}'
    return save_model_path
