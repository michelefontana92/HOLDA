from re import S
import ray
import torch
from utils.messages import CV_ValidationMessage, HoldOut_ValidationMessage
from utils.messages import ServerMessage
import datetime
from utils.metrics_sklearn import metrics_to_string
from metadata.meta import CheckPoint, CheckPoint
import numpy as np
import copy
import math
import os
from utils.util import create_model_name, create_model_name_state
import pickle as pkl


class Server:
    """
    # Description:

    It represents the central server of the hierarchy.
    It orchestrates the whole training process.
    Its children nodes could  be either a RemoteTrustedProxy or a RemoteLocalClient.
    It performs the Update-Aggregate-Evaluate loop.
    It stores the global model in its internal state.
    """

    def __init__(self, config, children_list):

        self.children_list = [child() for child in children_list]
        self.target_label = config.target_label
        self.use_deltas = config.use_deltas

        self.id = config.metadata.id
        self.log_file = config.metadata.log_path
        self.checkpoint_best_path = config.metadata.ckpt_best
        self.checkpoint_epoch_path = config.metadata.ckpt_epoch

        self.global_iter_id = 0
        self.state_id = 0
        self.save_all_models = False
        self.save_state_models = False
        self.save_all_models_path = config.metadata.save_all_models_path
        self.save_state_models_path = config.metadata.save_state_models_path
        self.history_path = f'{os.path.dirname(self.log_file)}/../History'
        if not os.path.exists(self.history_path):
            os.makedirs(self.history_path)
        if not config.metadata.save_all_models_path == '':
            path = create_model_name(
                config.metadata.save_all_models_path, 0, 0)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.save_all_models = True

        if not config.metadata.save_state_models_path == '':
            path = create_model_name_state(
                config.metadata.save_state_models_path, 0)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.save_state_models = True

        self.training_params = config.training_params
        self.global_epochs = self.training_params.epochs
        self.patience = self.training_params.patience
        self.sample_size = config.training_params.sample_size

        self.aggregation_fn = config.training_params.aggregation_fn
        self.init_model_fn = config.training_params.init_model_fn
        self.extract_model_fn = config.training_params.extract_model_fn

        self.epoch2ckpt = self.training_params.epoch2ckpt
        self.from_check = config.training_params.from_check
        assert len(
            children_list) > 0, f'[{self.id}]: The children list cannot be empty'

        log_dir = os.path.dirname(self.log_file)
        ckpt_best_dir = os.path.dirname(self.checkpoint_best_path)
        ckpt_epoch_dir = os.path.dirname(self.checkpoint_epoch_path)
        if not (log_dir == '') and not (os.path.exists(log_dir)):
            os.makedirs(log_dir)
        if not (ckpt_best_dir == '') and not (os.path.exists(ckpt_best_dir)):
            os.makedirs(ckpt_best_dir)
        if not (ckpt_epoch_dir == '') and not (os.path.exists(ckpt_epoch_dir)):
            os.makedirs(ckpt_epoch_dir)

        if config.training_params.starting_model_path == '':
            self.starting_model = None

        else:
            self.starting_model = torch.load(
                open(config.training_params.starting_model_path, 'rb'))

        self.val_from_file = True
        self.global_model = None
        self.server_msg = None

        self.current_train_weights = 0
        self.current_val_weights = 0

        self.total_train_weights = 0
        self.total_val_weights = 0
        self.best_epoch = 0
        self.selected_children = self.children_list
        self.n_global_model_updates = 0

        self.validation_msg = HoldOut_ValidationMessage()

        self.have_activated_children = False
        self.config = config

    def get_id(self):
        return self.id

    def eval_on_test(self):
        """
        # Description:

        It evaluates the model described in \(starting\_model\) on the test or validation set.

        # Returns:
        \(train\_metrics, val\_metrics\): (dict,dict)
            The evaluation scores obtained by the model on the training and validation data.

        """
        self.global_model = self.starting_model
        self.server_msg = ServerMessage(
            new_model=copy.deepcopy(self.global_model.state_dict()),
            validation_msg=self.validation_msg,
            send_deltas=self.use_deltas,
            target_label=self.target_label,
        )

        results = self.evaluate_metrics(self.server_msg)
        aggregated_result = self.aggregation_fn(results)
        train_metrics = aggregated_result.train_metrics
        val_metrics = aggregated_result.validation_metrics
        return train_metrics, val_metrics

    def init_computation(self):
        """
        # Description:

        It initializes the overall computation by calculating the training and validation weights.

        NOTE:
        It isn't necessary if the children are sampled at the beginning of each training round!
        """
        self.total_train_weights, self.total_val_weights = self.get_weights(
            self.validation_msg)

    def get_weights(self, message):
        """
        # Description:

        It computes the weights, i.e. the total number of records in the training and validation data.

        # Args:

        `message`: (ValidationMessage)
            A message containg the validation strategy that has to be perfomed, i.e k-foldCV or Hold-Out

        # Returns:

        \(train\_weights,val\_weights\): (float, float)
            The weights associated to the training and validation data.
        """
        train_weights = 0
        val_weights = 0
        handlers = [child.get_weights.remote(
            message) for child in self.selected_children]
        for handler in handlers:
            train_weight, val_weight = ray.get(handler)
            with open(self.log_file, 'a') as f:
                f.write(
                    f'{datetime.datetime.now()}: Get_Weights: '
                    f'Ricevuto: Training = {train_weight} '
                    f'Validation = {val_weight}\n')
            train_weights += train_weight
            val_weights += val_weight

        with open(self.log_file, 'a') as f:
            f.write(f'{datetime.datetime.now()}: Get_Weights: Pesi totali: '
                    f'Training = {train_weights} '
                    f'Validation = {val_weights}\n')
        return train_weights, val_weights

    def _sample_children(self, sample_size):
        """
        # Description:

        It randomly samples a fraction \(sample\_size \\in (0,1] \) of its direct children

        # Args:

        `sample_size`: (float \(\\in (0,1]\))
            The fraction of children to select

        """
        if sample_size == 1:
            self.selected_children = self.children_list
        else:
            n_samples = math.ceil(sample_size*len(self.children_list))
            self.selected_children = list(np.random.choice(
                self.children_list, size=n_samples, replace=False))

    def activate_children(self, mode):
        handlers = []
        for child in self.children_list:
            handlers.append(child.activate.remote(
                mode))
        for handler in handlers:
            ray.get(handler)

    def broadcast_train_msg(self, message):
        """
        # Description:

        It trasmits the message stored in `message` to all its selected children.

        # Args:

        `message` : (ServerMessage)
            The message to be transmitted to the children

        # Returns:

        `handlers` : (list of Future)
            A list containing the handlers needed to get back the result of the computation performed by each child.
            The length of the list is equal to the number of selected children.
        """
        handlers = []
        for child in self.selected_children:
            handlers.append(child.broadcast_train_msg.remote(
                message))
        return handlers

    def evaluate_metrics(self, message):
        """
        # Description:

        It evaluates the model stored in `message` according to the predefined metrics.
        The message is transmitted to every child node.

        # Args:

        `message` : (ServerMessage)
            The message to be transmitted to the children

        # Returns:

        `handlers` : (list of Future)
            A list containing the handlers needed to get back the result of the computation performed by each child.
            The length of the list is equal to the number of children.
        """
        handlers = []
        for child in self.children_list:
            handlers.append(child.evaluate_metrics.remote(
                message))
        return handlers

    def federated_training(self):
        """
        # Description:

        It executes the full training algorithm, i.e. HOLDA, on the selected training and validation data.

        # Returns:

        \(best\_train\_history, best\_val\_history\): (list, list)
            The training and validation history of each considered metric during the overall training process.
        """
        if self.from_check:
            ckpt = torch.load(open((f'{self.checkpoint_best_path}'
                                    ), 'rb'))

            train_history = ckpt.train_metrics
            val_history = ckpt.val_metrics
            best_val_f1 = ckpt.val_metrics['val_f1'][-1]
            self.global_model = self.training_params.build_model_fn()
            self.global_model.load_state_dict(ckpt.model)
            with open(self.log_file, 'a') as f:
                f.write(
                    f'{datetime.datetime.now()}: RESTART FROM THE LAST CHECKPOINT\n')
            epoch = len(val_history['val_f1'])
            self.best_epoch = epoch
            print('BEST F1: ', best_val_f1)
        else:
            train_history = {}
            val_history = {}

            best_val_f1 = -float('inf')
            self.global_model = self.training_params.build_model_fn()
            epoch = 0
            self.best_epoch = 0

        if self.starting_model:
            self.global_model = self.starting_model

        waiting_epochs = 0
        early_stop = False

        self.server_msg = ServerMessage(
            new_model=copy.deepcopy(self.global_model.state_dict()),
            validation_msg=self.validation_msg,
            send_deltas=self.use_deltas,
            target_label=self.target_label
        )

        self.server_msg.new_model = copy.deepcopy(
            self.global_model.state_dict())

        results = self.evaluate_metrics(self.server_msg)

        aggregated_result = self.aggregation_fn(results)
        train_metrics = aggregated_result.train_metrics
        val_metrics = aggregated_result.validation_metrics

        for metric, value in train_metrics.items():
            train_history[f'train_{metric}'] = [value]

        for metric, value in val_metrics.items():
            val_history[f'val_{metric}'] = [value]

        with open(self.log_file, 'a') as f:
            f.write(
                (f'{datetime.datetime.now()}: Epoch 0: '
                 f'TRAIN = {metrics_to_string(train_metrics)}\t'
                 f'VAL = {metrics_to_string(val_metrics)}\n')
            )

        if val_metrics['f1'] > best_val_f1:
            self.best_epoch = epoch
            best_val_f1 = val_metrics['f1']
            ckpt = CheckPoint(self.extract_model_fn(self.global_model),
                              train_history, val_history)
            torch.save(ckpt, open((f'{self.checkpoint_best_path}'), 'wb'))

            if self.save_state_models:
                path = create_model_name(
                    self.save_state_models_path, 0, epoch)
                torch.save(self.extract_model_fn(
                    self.global_model), open(path, 'wb'))
            del ckpt

            with open(self.log_file, 'a') as f:
                f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                         f'CHECKPOINT: Better Model Found\n'
                         ))

        while (not early_stop) and (epoch < self.global_epochs):
            self._sample_children(sample_size=self.sample_size)
            results = self.broadcast_train_msg(self.server_msg)

            aggregated_result = self.aggregation_fn(results)

            new_model = aggregated_result.new_model

            result_model = copy.deepcopy(new_model)

            if self.use_deltas:
                for key, value in self.global_model.state_dict().items():
                    result_model[key] = result_model[key] + value

            self.global_model.load_state_dict(result_model)
            self.server_msg.new_model = copy.deepcopy(result_model)

            results = self.evaluate_metrics(self.server_msg)

            aggregated_result = self.aggregation_fn(results)
            train_metrics = aggregated_result.train_metrics
            val_metrics = aggregated_result.validation_metrics

            if len(train_history) == 0:
                for metric, value in train_metrics.items():
                    train_history[f'train_{metric}'] = [value]

                for metric, value in val_metrics.items():
                    val_history[f'val_{metric}'] = [value]

            else:
                for metric, value in train_metrics.items():
                    train_history[f'train_{metric}'].append(value)

                for metric, value in val_metrics.items():
                    val_history[f'val_{metric}'].append(value)

            with open(self.log_file, 'a') as f:
                f.write(
                    (f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                        f'TRAIN = {metrics_to_string(train_metrics)}\t'
                        f'VAL = {metrics_to_string(val_metrics)}\n')
                )

            if self.save_all_models:
                path = create_model_name(
                    self.save_all_models_path, 0, epoch)
                torch.save(self.extract_model_fn(
                    self.global_model), open(path, 'wb'))

            if val_metrics['f1'] > best_val_f1:
                waiting_epochs = 0
                self.best_epoch = epoch
                best_val_f1 = val_metrics['f1']
                ckpt = CheckPoint(self.extract_model_fn(self.global_model),
                                  train_history, val_history,
                                  )
                torch.save(ckpt, open((f'{self.checkpoint_best_path}'), 'wb'))

                if self.save_state_models:
                    path = create_model_name_state(
                        self.save_state_models_path, self.state_id)
                    torch.save(self.extract_model_fn(
                        self.global_model), open(path, 'wb'))
                    self.state_id += 1
                del ckpt

                with open(self.log_file, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                             f'CHECKPOINT: Better Model Found\n'
                             ))

            else:
                waiting_epochs += 1
                if waiting_epochs % self.patience == 0:
                    early_stop = True
                    with open(self.log_file, 'a') as f:
                        f.write((f'{datetime.datetime.now()}:'
                                 'EARLY STOPPING\n'
                                 ))

            if epoch % self.epoch2ckpt == 0:
                ckpt = CheckPoint(self.extract_model_fn(self.global_model),
                                  train_history, val_history,
                                  )

                torch.save(ckpt, open((f'{self.checkpoint_epoch_path}'), 'wb'))

                del ckpt

                with open(self.log_file, 'a') as f:
                    f.write((f'{datetime.datetime.now()}:'
                             f' Epoch {epoch + 1}: '
                             f'CHECKPOINT: Passed '
                             f'{self.epoch2ckpt} epochs\n'
                             ))
            epoch += 1

        ckpt = torch.load(open((f'{self.checkpoint_best_path}'), 'rb'))
        best_model = ckpt.model
        best_train_history = ckpt.train_metrics
        best_val_history = ckpt.val_metrics

        self.global_model = self.init_model_fn(self.global_model, best_model)
        torch.save(self.global_model,
                   f'{os.path.dirname(self.checkpoint_best_path)}/global_model.h5')
        del ckpt
        return best_train_history, best_val_history

    def _extend_history(self, history_list):
        max_len = 0
        for history in history_list:
            max_len = max(max_len, len(history))

        mean_hist = np.zeros((len(history_list), max_len))
        for i, history in enumerate(history_list):
            mean_hist[i, :len(history)] = history
            mean_hist[i, len(history):] = history[-1]

        mean_hist = np.round(np.mean(mean_hist, axis=0), 3)
        return mean_hist

    def execute(self):
        """
        # Description:

        The main entry point of HOLDA.
        The function starts the whole training process of HOLDA.

        # Returns:

        `(train_history, val_history): (dict, dict)`
        The history with the scores about the metrics considered, computed on the training and validation data, respectively.
        A general entry of the dictionary is "{'metric_name': [v_1,v_2,...v_k]}"

        NOTE:
            The k-foldCV process has still to be tested!

        """
        with open(self.log_file, 'w') as f:
            f.write(f'{datetime.datetime.now()}: {self.id} ATTIVATO\n')
            f.write(f'Training: Eseguo {self.global_epochs} epoche globali '
                    f'Early stopping patience = {self.patience} epoche\n'
                    f'Training params: {self.config.training_params}\n')
            children_id = [ray.get(child.get_id.remote())
                           for child in self.children_list]
            f.write(f'Children list : {children_id}\n')

        self.activate_children('w')
        self.have_activated_children = True
        if isinstance(self.validation_msg, CV_ValidationMessage):
            total_folds = self.validation_msg.total_folds

            train_folds_hist = {}
            val_folds_hist = {}

            for fold in range(self.validation_msg.total_folds):
                with open(self.log_file, 'a') as f:
                    f.write(f'Starting Fold {fold+1} / {total_folds}\n')

                self.validation_msg.current_fold = fold

                train_history, val_history = self.federated_training()

                current_train_result = {}
                current_val_result = {}

                for metric, values in train_history.items():
                    current_train_result[metric] = values[-1]
                    if fold == 0:
                        train_folds_hist[metric] = [values]

                    else:
                        train_folds_hist[metric].append(values)

                for metric, values in val_history.items():
                    current_val_result[metric] = values[-1]
                    if fold == 0:
                        val_folds_hist[metric] = [values]
                    else:
                        val_folds_hist[metric].append(values)

                with open(self.log_file, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: '
                             f'Ended Fold {fold+1} / {total_folds}: '
                             f'TRAIN = '
                             f'{metrics_to_string(current_train_result)}\t'
                             f'VAL = '
                             f'{metrics_to_string(current_val_result)}\n\n'))

            # Ho terminato di esaminare tutti i folds
            for metric, values in train_folds_hist.items():
                train_folds_hist[metric] = self._extend_history(
                    train_folds_hist[metric])

            for metric, values in val_folds_hist.items():
                val_folds_hist[metric] = self._extend_history(
                    val_folds_hist[metric])

            train_cv_result = {}
            val_cv_result = {}

            for metric, values in train_folds_hist.items():
                train_cv_result[metric] = values[-1]
            for metric, values in val_folds_hist.items():
                val_cv_result[metric] = values[-1]

            with open(self.log_file, 'a') as f:
                f.write((f'{datetime.datetime.now()}: '
                         f'Ended CV: FINAL RESULTS: '
                         f'TRAIN = {metrics_to_string(train_cv_result)}\t'
                         f'VAL = {metrics_to_string(val_cv_result)}\n\n'))

            return train_folds_hist, val_folds_hist

        else:  # hold-out strategy

            train_history, val_history = self.federated_training()
            train_result = {}
            val_result = {}
            for metric, values in train_history.items():
                train_result[metric] = values[-1]
            for metric, values in val_history.items():
                val_result[metric] = values[-1]

            with open(self.log_file, 'a') as f:
                f.write((f'{datetime.datetime.now()}: '
                         f'Ended HOLD OUT: FINAL RESULTS: '
                         f'TRAIN = {metrics_to_string(train_result)}\t'
                         f'VAL = {metrics_to_string(val_result)}\n\n'))

            return train_history, val_history

    def personalize(self):
        with open(self.log_file, 'a') as f:
            f.write(
                f'\n\n{datetime.datetime.now()}: '
                f'Starting the personalization phase!\n')
        if not self.have_activated_children:
            self.activate_children('a')
            self.global_model = self.training_params.build_model_fn()
            self.server_msg = ServerMessage(
                new_model=copy.deepcopy(self.global_model.state_dict()),
                validation_msg=self.validation_msg,
                send_deltas=self.use_deltas,
                target_label=self.target_label
            )

        handlers = [child.personalize.remote(self.server_msg)
                    for child in self.children_list]
        for handler in handlers:
            ray.get(handler)
        return

    def shutdown(self):
        """
        It starts the shutdown phase, which frees the resources and closes the federation.
        """
        handlers = [child.shutdown.remote()
                    for child in self.children_list]
        for handler in handlers:
            ray.get(handler)

        ckpt = torch.load(open(self.checkpoint_best_path, 'rb'))
        train_metrics = {}
        val_metrics = {}
        for metric in ckpt.train_metrics.keys():
            train_metrics[metric] = ckpt.train_metrics[metric][-1]
            pkl.dump(ckpt.train_metrics[metric],
                     file=open(
                         f'{self.history_path}/{self.id}_{metric}_hist.pkl', 'wb'))

        for metric in ckpt.val_metrics.keys():
            val_metrics[metric] = ckpt.val_metrics[metric][-1]
            pkl.dump(ckpt.val_metrics[metric],
                     file=open(
                         f'{self.history_path}/{self.id}_{metric}_hist.pkl', 'wb'))

        with open(self.log_file, 'a') as f:
            f.write('\n')
            f.write(200*'-')
            f.write('\n')
            f.write(f'{datetime.datetime.now()}: Starting the shutdown...\n')
            f.write(f'Best model: {metrics_to_string(train_metrics)}\t'
                    f'{metrics_to_string(val_metrics)}\n')
            f.write(200*'-')
            f.write('\n')
            f.write(200*'-')
            f.write('\n')
            f.write((f'{datetime.datetime.now()}: '
                     f'Shutdown completed\n'))
