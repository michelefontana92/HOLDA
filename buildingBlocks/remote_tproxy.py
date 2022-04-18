from buildingBlocks.server import Server
import ray
import copy
from utils.messages import ClientMessage, ServerMessage
import datetime
import math
from utils.metrics_sklearn import metrics_to_string
import pickle as pkl
import os
from metadata.meta import CheckPoint
import torch


@ray.remote(num_cpus=1)
class RemoteTrustedProxy(Server):
    """
    # Description:

    It represents an intermediate server, also called proxy, of the hierarchy.
    It orchestrates the training process in the sub-hierarchy rooted in itself.
    Its children nodes could  be either a RemoteTrustedProxy or a RemoteLocalClient.
    It performs the Update-Aggregate-Evaluate loop.
    It stores the best generalizing model in its internal state.
    """

    def __init__(self, config, children_list):
        super().__init__(config, children_list)

        self.epoch = 0
        self.only_eval = True

        if self.from_check:
            ckpt = torch.load(open((f'{self.checkpoint_best_path}'
                                    ), 'rb'))
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
            self.train_history = {}
            self.val_history = {}
            self.best_f1 = -float('inf')
            self.epoch = 0

    def activate(self, mode):
        self.activate_children('w')
        with open(self.log_file, 'w') as f:
            f.write(f'{datetime.datetime.now()}: {self.id} ACTIVATED\n')
            f.write(f'Training: I execute {self.global_epochs} global epochs '
                    f'Early stopping patience = {self.patience} epochs\n'
                    f'Training params: {self.config.training_params}\n')
            children_id = [ray.get(child.get_id.remote())
                           for child in self.children_list]
            f.write(f'Children list : {children_id}\n')

    def broadcast_train_msg(self, message):
        """
        # Description:
        It executes HOLDA for a given number of epochs.
        As the server, it trasmits the message stored in `message` to all its selected children.
        If `self.send_last\_model` == False, it sends back to the parent node the parameters of the best model

        # Args:

        `message` : (ServerMessage)
            The message to be transmitted to the children

        # Returns:

        `message_result` : ([ClientMessage]])
            A list containing the message that has to be transmitted to the parent node.
        """
        self.only_eval = False
        self.epoch += 1

        train_history = {}
        val_history = {}
        self.global_model = self.training_params.build_model_fn()
        with open(self.log_file, 'a') as f:
            f.write(
                f'{datetime.datetime.now()}: '
                f'EPOCH {self.epoch}\n')
        current_total_train_weight = 0
        current_total_val_weight = 0

        self.global_model.load_state_dict(copy.deepcopy(message.new_model))
        initial_model = copy.deepcopy(message.new_model)

        new_message = ServerMessage(
            new_model=None,
            validation_msg=message.validation_msg,
            send_deltas=message.send_deltas,
            target_label=message.target_label
        )

        for epoch in range(self.global_epochs):
            self._sample_children(sample_size=self.sample_size)
            new_message.new_model = copy.deepcopy(
                self.global_model.state_dict())
            handlers = []

            for child in self.selected_children:
                handlers.append(child.broadcast_train_msg.remote(
                    new_message))

            aggregated_result = self.aggregation_fn(handlers)
            new_model = aggregated_result.new_model
            result_model = copy.deepcopy(new_model)

            if self.use_deltas:
                for key, value in self.global_model.state_dict().items():
                    result_model[key] = result_model[key] + value

            self.global_model.load_state_dict(result_model)

            handlers = []

            for child in self.selected_children:
                new_message.new_model = copy.deepcopy(
                    result_model)
                handlers.append(child.evaluate_metrics.remote(
                    new_message))

            aggregated_result = self.aggregation_fn(handlers)
            train_metrics = aggregated_result.train_metrics
            val_metrics = aggregated_result.validation_metrics

            current_total_train_weight += aggregated_result.train_weight
            current_total_val_weight += aggregated_result.validation_weight

            with open(self.log_file, 'a') as f:
                f.write(
                    (f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                        f'TRAIN = {metrics_to_string(train_metrics)}\t'
                        f'VAL = {metrics_to_string(val_metrics)}\n')
                )

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

            if val_metrics['f1'] > self.best_f1:
                self.best_epoch = epoch
                self.best_f1 = val_metrics['f1']
                ckpt = CheckPoint(self.extract_model_fn(self.global_model),
                                  train_history, val_history)
                torch.save(ckpt, open((f'{self.checkpoint_best_path}'), 'wb'))

                del ckpt

                with open(self.log_file, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                             f'CHECKPOINT: Better Model Found\n'
                             ))

        del new_message

        current_total_train_weight = math.ceil(
            current_total_train_weight / self.global_epochs)
        current_total_val_weight = math.ceil(
            current_total_val_weight / self.global_epochs)

        ckpt = torch.load(open((f'{self.checkpoint_best_path}'), 'rb'))
        best_train_history = ckpt.train_metrics
        best_val_history = ckpt.val_metrics
        result_model = ckpt.model

        del ckpt

        if self.use_deltas:
            for key, value in initial_model.items():
                result_model[key] = result_model[key] - value

        self.global_model = self.init_model_fn(self.global_model, result_model)
        train_result = {}
        val_result = {}

        for metric, values in best_train_history.items():
            train_result[metric] = values[-1]
        for metric, values in best_val_history.items():
            val_result[metric] = values[-1]

        with open(self.log_file, 'a') as f:
            f.write(
                (f'{datetime.datetime.now()}: ENDED TRAINING: '
                 f'Best epoch {self.best_epoch + 1}: '
                 f'TRAIN = {metrics_to_string(train_result)}\t'
                 f'VAL = {metrics_to_string(val_result)}\n\n')
            )

        self.global_iter_id += 1

        message_result = ClientMessage(
            result_model,
            train_metrics,
            val_metrics,
            current_total_train_weight,
            current_total_val_weight,
        )

        return [message_result]

    def evaluate_metrics(self, message):
        """
        # Description:

        It evaluates the model stored in `message` according to the predefined metrics.
        The message is transmitted to every child node.

        # Args:

        `message` : (ServerMessage)
            The message to be transmitted to the children

        # Returns:

        `aggregated_result` : (list of dict)
            A list containing the result of the evaluation for each considered metric.
        """
        handlers = []
        for child in self.children_list:
            handlers.append(child.evaluate_metrics.remote(
                message))

        aggregated_result = self.aggregation_fn(handlers)
        train_metrics = aggregated_result.train_metrics
        val_metrics = aggregated_result.validation_metrics
        with open(self.log_file, 'a') as f:
            f.write(
                (f'{datetime.datetime.now()}: Evaluating global model: '
                 f'TRAIN = {metrics_to_string(train_metrics)}\t'
                 f'VAL = {metrics_to_string(val_metrics)}\n')
            )
        if len(self.train_history) == 0:
            for metric, value in train_metrics.items():
                self.train_history[f'train_{metric}'] = [value]

            for metric, value in val_metrics.items():
                self.val_history[f'val_{metric}'] = [value]

        else:
            for metric, value in train_metrics.items():
                self.train_history[f'train_{metric}'].append(value)

            for metric, value in val_metrics.items():
                self.val_history[f'val_{metric}'].append(value)

            ckpt = CheckPoint(message.new_model,
                              self.train_history, self.val_history)
            torch.save(ckpt, open(f'{self.checkpoint_best_path}', 'wb'))

            del ckpt

            with open(self.log_file, 'a') as f:
                f.write((f'{datetime.datetime.now()}: Epoch {self.epoch}: '
                         f'CHECKPOINT: Better Model Found\n'
                         ))

        return [aggregated_result]

    def _pers_training(self, message):
        ckpt = torch.load(open((f'{self.checkpoint_best_path}'), 'rb'))
        train_history = ckpt.train_metrics
        val_history = ckpt.val_metrics

        self.global_model = self.training_params.build_model_fn()
        self.global_model.load_state_dict(copy.deepcopy(ckpt.model))
        new_message = ServerMessage(
            new_model=None,
            validation_msg=message.validation_msg,
            send_deltas=message.send_deltas,
            target_label=message.target_label
        )

        for epoch in range(self.global_epochs):
            self._sample_children(sample_size=self.sample_size)
            new_message.new_model = copy.deepcopy(
                self.global_model.state_dict())
            handlers = []

            for child in self.selected_children:
                handlers.append(child.broadcast_train_msg.remote(
                    new_message))

            aggregated_result = self.aggregation_fn(handlers)
            new_model = aggregated_result.new_model
            result_model = copy.deepcopy(new_model)

            if self.use_deltas:
                for key, value in self.global_model.state_dict().items():
                    result_model[key] = result_model[key] + value

            self.global_model.load_state_dict(result_model)

            handlers = []

            for child in self.selected_children:
                new_message.new_model = copy.deepcopy(
                    result_model)
                handlers.append(child.evaluate_metrics.remote(
                    new_message))

            aggregated_result = self.aggregation_fn(handlers)
            train_metrics = aggregated_result.train_metrics
            val_metrics = aggregated_result.validation_metrics

            with open(self.log_file, 'a') as f:
                f.write(
                    (f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                        f'TRAIN = {metrics_to_string(train_metrics)}\t'
                        f'VAL = {metrics_to_string(val_metrics)}\n')
                )

            if val_metrics['f1'] > self.best_f1:
                self.best_epoch = epoch
                self.best_f1 = val_metrics['f1']

                for metric, value in train_metrics.items():
                    train_history[f'train_{metric}'].append(value)

                for metric, value in val_metrics.items():
                    val_history[f'val_{metric}'].append(value)

                ckpt = CheckPoint(self.extract_model_fn(self.global_model),
                                  train_history, val_history)
                torch.save(ckpt, open((f'{self.checkpoint_best_path}'), 'wb'))

                del ckpt

                with open(self.log_file, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                             f'CHECKPOINT: Better Model Found\n'
                             ))
        return

    def personalize(self, server_message):
        with open(self.log_file, 'a') as f:
            f.write(
                f'\n\n{datetime.datetime.now()}: '
                f'Starting the personalization phase!\n')
        if not self.have_activated_children:
            self.activate_children('a')
            self.global_model = self.training_params.build_model_fn()

        self._pers_training(server_message)
        with open(self.log_file, 'a') as f:
            f.write(
                f'\n\n{datetime.datetime.now()}: '
                f'Starting the chldren personalization phase!\n')
        handlers = [child.personalize.remote(server_message)
                    for child in self.children_list]
        for handler in handlers:
            ray.get(handler)
        return

    def shutdown(self):

        if self.only_eval:
            handlers = [child.shutdown.remote()
                        for child in self.children_list]

            for handler in handlers:
                ray.get(handler)
            with open(self.log_file, 'a') as f:
                f.write((f'{datetime.datetime.now()}: '
                         f'Shutdown completed\n'))
        else:
            ckpt = torch.load(open(f'{self.checkpoint_best_path}', 'rb'))

            best_train_history = ckpt.train_metrics
            best_val_history = ckpt.val_metrics

            train_result = {}
            val_result = {}

            for metric, values in best_train_history.items():
                train_result[metric] = values[-1]
            for metric, values in best_val_history.items():
                val_result[metric] = values[-1]

            for metric, value in best_train_history.items():
                pkl.dump(value,
                         file=open(
                             f'{self.history_path}/{self.id}_{metric}_hist.pkl', 'wb'))

            for metric, value in best_val_history.items():
                pkl.dump(value,
                         file=open(
                             f'{self.history_path}/{self.id}_{metric}_hist.pkl', 'wb'))

            handlers = [child.shutdown.remote()
                        for child in self.children_list]

            for handler in handlers:
                ray.get(handler)

            with open(self.log_file, 'a') as f:
                f.write('\n')
                f.write(200*'-')
                f.write('\n')
                f.write(f'{datetime.datetime.now()}: Starting the shutdown...\n')
                f.write(f'Best model: {metrics_to_string(train_result)}\t'
                        f'{metrics_to_string(val_result)}\n')
                f.write(200*'-')
                f.write('\n')
                f.write(200*'-')
                f.write('\n')
                f.write((f'{datetime.datetime.now()}: '
                         f'Shutdown completed\n'))
