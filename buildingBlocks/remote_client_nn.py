from buildingBlocks.remote_client import RemoteLocalClient
from utils.data_loaders import create_data_loader
import torch.nn as nn
import torch
from utils.messages import CV_ValidationMessage, ClientMessage
from utils.messages import ClientMessage
import datetime
from utils.metrics_sklearn import evaluate_metrics, metrics_to_string
import numpy as np
from metadata.meta import CheckPoint
import pandas as pd
import ray
import copy
import pickle as pkl
import os

from utils.util import create_model_name, create_model_name_monitor, create_model_name_state


@ray.remote(num_cpus=1)
class RemoteLocalClient_NN(RemoteLocalClient):
    """
    # Description:

    It represents a Local Client, which trains a Neural Network in a federated way.
    It is a subclass of `RemoteLocalClient`

    """

    def __init__(self, config):
        super().__init__(config)
        self.build_model_fn = config.training_params.build_model_fn
        self.optimizer_fn = config.training_params.optimizer_fn
        self.criterion_fn = config.training_params.loss_fn
        self.n_epochs = config.training_params.epochs
        self.batch_size = config.training_params.batch_size
        self.patience = config.training_params.patience
        self.use_weights = config.training_params.use_weights
        self.epoch2ckpt = config.training_params.epoch2ckpt

    def _predict(self, data_loader, model):
        y_true = np.array([])
        y_pred = np.array([])

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(data_loader):
                data = batch['data']
                labels = batch['target']
                outputs = self.model(data)
                _, predicted = torch.max(torch.log_softmax(outputs, 1), 1)

                y_true = np.append(y_true, labels.flatten().detach().numpy())
                y_pred = np.append(
                    y_pred, predicted.flatten().detach().numpy())
        return y_pred

    def _evaluate_model(self, data_loader, compute_loss=False):
        """
        # Description:
        It evaluates the current model, stored in self.model, according to the predefined metrics.

        # Args:

        `data_loader`: (DataLoader)
        Pytorch data loader of the underlying dataset

        `compute_loss`: (bool)
        If True, the function computes the value of the loss.
        If False, the value of the loss returned in output is set to 0.0
        # Returns:
        `(loss, metrics) : (float, dict)`
        The value of the loss function (if computed) and the score for each selected metric.
        """
        total_loss = 0.0
        n_batches = 0

        if compute_loss:
            criterion = self.criterion_fn()

        y_true = np.array([])
        y_pred = np.array([])

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(data_loader):
                n_batches += 1
                data = batch['data']
                labels = batch['target']
                outputs = self.model(data)
                _, predicted = torch.max(torch.log_softmax(outputs, 1), 1)

                y_true = np.append(y_true, labels.flatten().detach().numpy())
                y_pred = np.append(
                    y_pred, predicted.flatten().detach().numpy())

                if compute_loss:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

            if compute_loss:
                total_loss = round(total_loss / n_batches, 4)

            metrics = evaluate_metrics(
                self.metrics, y_true, y_pred)

        del y_true
        del y_pred

        return total_loss, metrics

    def _build_train_and_val_set(self, server_message):
        """
        # Description:

        The function builds the training and the validation sets that have to be used during the training process.

        # Args:

        `server_message`: (ServerMessage)
            Message received from the parent node.

        # Returns:

        `(train_set, val_set): (DataFrame, DataFrame)`
        The training and the validation set, respectively


        """

        if server_message.validation_msg.from_file:
            train_set = pd.read_csv(self.train_path)
            val_set = pd.read_csv(self.val_path)
        else:
            self.dataset = pd.read_csv(self.dataset_path)

            if len(self.fold_split) == 0:
                self._split_dataset(
                    self.dataset, server_message.validation_msg)

            if isinstance(server_message.validation_msg, CV_ValidationMessage):
                current_fold = server_message.validation_msg.current_fold
                train_set = self.dataset.iloc[self.fold_split[current_fold][0]]
                val_set = self.dataset.iloc[self.fold_split[current_fold][1]]
                with open(self.log_path, 'a') as f:
                    f.write(
                        f'{datetime.datetime.now()}: '
                        f'KCV: Eseguo la validazione sul fold '
                        f'{current_fold+1}/{len(self.fold_split)}\n')

            else:
                train_set = self.dataset.iloc[self.fold_split[0][0]]
                val_set = self.dataset.iloc[self.fold_split[0][1]]
        return train_set, val_set

    def broadcast_train_msg(self, server_message):
        """
        # Description:

        It performs the local training of the received model and eventually updates the internal state of the local client.
        It trains the model, starting from the received parameters, up to the given number of epochs.
        The early stopping is employed as stopping condition.
        The function sends back the parameter of the best generalizing model to the parent node, i.e. the parameters stored in the internal state.

        # Args:

        `server_message`: (ServerMessage)
        The message received from the parent node.

        # Returns:

        # [message: ClientMessage]

        A list with just one element. The message which is transmitted to the parent node, containing the updated model parameters and the number of records in the training and validation set.
        """
        with open(self.log_path, 'a') as f:
            f.write(
                f'\n{datetime.datetime.now()}: '
                f'Arrivato nuovo modello dal server\n')

        self.model = self.build_model_fn()
        self.send_deltas = server_message.send_deltas
        initial_model_dict = copy.deepcopy(server_message.new_model)
        self.model.load_state_dict(copy.deepcopy(server_message.new_model))

        self.optimizer = self.optimizer_fn(self.model.parameters())
        best_val_f1 = -float('inf')
        waiting_epochs = 0
        early_stopping = False
        train_set = None
        val_set = None

        path = create_model_name_monitor(
            f'{os.path.dirname(self.ckpt_best)}/{self.id}.pt',
            self.global_iter_id, input=True)
        torch.save(self.model.state_dict(), open(path, 'wb'))

        train_set, val_set = self._build_train_and_val_set(server_message)

        with open(self.log_path, 'a') as f:
            f.write(
                f'{datetime.datetime.now()}: '
                f'Dimensione del Training set: '
                f'{len(train_set)} '
                f'Dimensione del Validation set: '
                f'{len(val_set)}\n')

        if self.use_weights:
            class_weights = self._compute_class_weights(
                train_set, server_message.target_label)

            criterion = self.criterion_fn(weight=class_weights)
        else:
            criterion = self.criterion_fn()

        train_loader = create_data_loader(
            train_set, server_message.target_label, self.batch_size, shuffle=True)

        val_loader = create_data_loader(
            val_set, server_message.target_label, self.batch_size, shuffle=True)

        epoch = 0

        while (not early_stopping) and (epoch < self.n_epochs):
            total_loss = 0.0
            num_batches = 0
            self.model.train()
            for _, batch in enumerate(train_loader):
                num_batches += 1
                data = batch['data']
                labels = batch['target']

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_loss = round(total_loss / num_batches, 4)

            _, train_metrics = self._evaluate_model(
                train_loader, compute_loss=False)

            val_loss, val_metrics = self._evaluate_model(
                val_loader, compute_loss=True)

            train_metrics['loss'] = total_loss
            val_metrics['loss'] = val_loss

            with open(self.log_path, 'a') as f:
                f.write((f'{datetime.datetime.now()}: Epoch {epoch+1}: '
                         f'TRAIN: {metrics_to_string(train_metrics)}\t'
                         f'VL:{metrics_to_string(val_metrics)}\n'))

            val_f1 = val_metrics['f1']

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                waiting_epochs = 0

                if val_f1 > self.state_best_f1:
                    self.state_best_f1 = val_f1

                    chkpoint = CheckPoint(
                        self.model.state_dict(),
                        train_metrics,
                        val_metrics,
                    )
                    torch.save(chkpoint, open(
                        self.ckpt_best, 'wb'))

                    del chkpoint
                    with open(self.log_path, 'a') as f:
                        f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                                 f'CHECKPOINT: Better Model Found\n'
                                 ))

            else:
                waiting_epochs += 1
                if waiting_epochs >= self.patience:
                    early_stopping = True
            if epoch % self.epoch2ckpt == 0:
                chkpoint = CheckPoint(
                    self.model.state_dict(),
                    train_metrics,
                    val_metrics,

                )

                torch.save(chkpoint, open(
                    self.ckpt_epoch, 'wb'))
                del chkpoint

                with open(self.log_path, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                             f'CHECKPOINT: Passed '
                             f'{self.epoch2ckpt} epochs\n'
                             ))

            epoch += 1

        if early_stopping:
            with open(self.log_path, 'a') as f:
                f.write((f'{datetime.datetime.now()}: Epoch {epoch}: '
                         f'EARLY STOPPING\n'
                         ))

        ckpt = torch.load(open(self.ckpt_best, 'rb'))
        result_model = ckpt.model

        path = create_model_name_monitor(
            f'{os.path.dirname(self.ckpt_best)}/{self.id}.pt',
            self.global_iter_id, input=False)
        torch.save(result_model, open(path, 'wb'))

        if self.send_deltas:
            for key, value in initial_model_dict.items():
                result_model[key] = result_model[key] - value

        self.current_iteration += 1
        self.global_iter_id += 1

        for metric, value in ckpt.train_metrics.items():
            self.train_history_out[f'train_{metric}'].append(value)
        for metric, value in ckpt.val_metrics.items():
            self.val_history_out[f'val_{metric}'].append(value)

        client_msg = ClientMessage(
            result_model,
            ckpt.train_metrics,
            ckpt.val_metrics,
            len(train_set),
            len(val_set)
        )

        del ckpt
        train_metrics = client_msg.train_metrics
        val_metrics = client_msg.validation_metrics

        with open(self.log_path, 'a') as f:
            f.write((f'{datetime.datetime.now()}: Ended Training: '
                     f'TRAIN: {metrics_to_string(train_metrics)}\t'
                     f'VL:{metrics_to_string(val_metrics)}\n\n'))

        del self.model
        del self.optimizer
        if server_message.validation_msg.from_file:
            del train_set
            del val_set
        else:
            del self.dataset
        return [client_msg]

    def evaluate_metrics(self, server_message):
        """
        # Description:

        It performs the evaluation of the received model according to the predefined metrics. It eventually updates the internal state of the local client.

        # Args:

        `server_message`: (ServerMessage)
        The message received from the parent node.

        # Returns:

        # [message: ClientMessage]

        A list with just one element. It stores the message which is transmitted to the parent node, made up of the model evaluation scores.

        """
        self.model = self.build_model_fn()
        self.model.load_state_dict(server_message.new_model)

        train_set, val_set = self._build_train_and_val_set(server_message)
        train_loader = create_data_loader(
            train_set, server_message.target_label, self.batch_size, shuffle=True)

        val_loader = create_data_loader(
            val_set, server_message.target_label, self.batch_size, shuffle=True)

        train_loss, train_metrics = self._evaluate_model(
            train_loader, compute_loss=True)

        val_loss, val_metrics = self._evaluate_model(
            val_loader, compute_loss=True)

        train_metrics['loss'] = train_loss
        val_metrics['loss'] = val_loss

        with open(self.log_path, 'a') as f:
            f.write((f'{datetime.datetime.now()}: Evaluating global model: '
                     f'TRAIN: {metrics_to_string(train_metrics)}\t'
                     f'VL:{metrics_to_string(val_metrics)}\n'))
        if val_metrics['f1'] > self.state_best_f1:
            with open(self.log_path, 'a') as f:
                f.write((f'{datetime.datetime.now()}: '
                         f'CHECKPOINT: Better Model Found\n'
                         ))
            self.state_best_f1 = val_metrics['f1']

            chkpoint = CheckPoint(
                self.model.state_dict(),
                train_metrics,
                val_metrics,
            )
            torch.save(chkpoint,
                       (self.ckpt_best))

        client_msg = ClientMessage(
            server_message.new_model,
            train_metrics,
            val_metrics,
            len(train_set),
            len(val_set),

        )
        return [client_msg]

    def personalize(self, server_message):
        self.n_epochs = self.config.pers_training_params.epochs
        self.optimizer_fn = self.config.pers_training_params.optimizer_fn
        self.criterion_fn = self.config.pers_training_params.loss_fn
        self.batch_size = self.config.pers_training_params.batch_size
        self.patience = self.config.pers_training_params.patience
        self.use_weights = self.config.pers_training_params.use_weights
        self.epoch2ckpt = self.config.pers_training_params.epoch2ckpt

        ckpt = torch.load(open(self.ckpt_best, 'rb'))
        self.model = self.build_model_fn()
        self.model.load_state_dict(copy.deepcopy(ckpt.model))

        self.state_id = 0
        self.optimizer = self.optimizer_fn(self.model.parameters())

        with open(self.log_path, 'a') as f:
            f.write(
                f'\n\n{datetime.datetime.now()}: '
                f'Personalizzo il miglior modello!\n')

        best_val_f1 = ckpt.val_metrics['f1']

        waiting_epochs = 0
        early_stopping = False
        train_set = None
        val_set = None

        train_set, val_set = self._build_train_and_val_set(server_message)

        if self.use_weights:
            class_weights = self._compute_class_weights(
                train_set, server_message.target_label)

            criterion = self.criterion_fn(weight=class_weights)
        else:
            criterion = self.criterion_fn()

        train_loader = create_data_loader(
            train_set, server_message.target_label,
            self.batch_size,
            shuffle=True)

        val_loader = create_data_loader(
            val_set, server_message.target_label,
            self.batch_size,
            shuffle=True)

        epoch = 0

        while (not early_stopping) and (epoch < self.n_epochs):
            total_loss = 0.0
            num_batches = 0
            self.model.train()
            for i, batch in enumerate(train_loader):
                num_batches += 1
                data = batch['data']
                labels = batch['target']

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_loss = round(total_loss / num_batches, 4)

            _, train_metrics = self._evaluate_model(
                train_loader, compute_loss=False)

            val_loss, val_metrics = self._evaluate_model(
                val_loader, compute_loss=True)

            train_metrics['loss'] = total_loss
            val_metrics['loss'] = val_loss

            with open(self.log_path, 'a') as f:
                f.write((f'{datetime.datetime.now()}: Epoch {epoch+1}: '
                         f'TRAIN: {metrics_to_string(train_metrics)}\t'
                         f'VL:{metrics_to_string(val_metrics)}\n'))

            val_f1 = val_metrics['f1']

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                waiting_epochs = 0

                chkpoint = CheckPoint(
                    self.model.state_dict(),
                    train_metrics,
                    val_metrics
                )
                torch.save(chkpoint, open(
                    self.ckpt_best, 'wb'))

                with open(self.log_path, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                             f'CHECKPOINT: Better Model Found\n'
                             ))

            else:
                waiting_epochs += 1
                if waiting_epochs >= self.patience:
                    early_stopping = True
            if epoch % self.epoch2ckpt == 0:
                chkpoint = CheckPoint(
                    self.model.state_dict(),
                    train_metrics,
                    val_metrics,
                )

                torch.save(chkpoint, open(
                    self.ckpt_epoch, 'wb'))
                del chkpoint

                with open(self.log_path, 'a') as f:
                    f.write((f'{datetime.datetime.now()}: Epoch {epoch + 1}: '
                             f'CHECKPOINT: Passed '
                             f'{self.epoch2ckpt} epochs\n'
                             ))

            epoch += 1
        if early_stopping:
            with open(self.log_path, 'a') as f:
                f.write((f'{datetime.datetime.now()}: Epoch {epoch}: '
                         f'EARLY STOPPING\n'
                         ))

        ckpt = torch.load(open(self.ckpt_best, 'rb'))
        result_model = ckpt.model
        for metric, value in ckpt.train_metrics.items():
            self.train_history_out[f'train_{metric}'].append(value)
        for metric, value in ckpt.val_metrics.items():
            self.val_history_out[f'val_{metric}'].append(value)
        path = create_model_name_monitor(
            f'{os.path.dirname(self.ckpt_best)}/{self.id}.pt',
            self.global_iter_id, input=False)
        torch.save(result_model, open(path, 'wb'))

        return

    def shutdown(self):
        """
        # Description:

        It shuts down the local client
        """
        final_train_metrics = {}
        final_val_metrics = {}
        for metric in self.train_history_out.keys():
            final_train_metrics[metric] = self.train_history_out[metric][-1]
            pkl.dump(self.train_history_out[metric],
                     file=open(
                f'{self.history_path}/{self.id}_{metric}_hist.pkl', 'wb'))

        for metric in self.val_history_out.keys():
            final_val_metrics[metric] = self.val_history_out[metric][-1]
            pkl.dump(self.val_history_out[metric],
                     file=open(
                f'{self.history_path}/{self.id}_{metric}_hist.pkl', 'wb'))

        with open(self.log_path, 'a') as f:
            f.write('\n')
            f.write(200*'-')
            f.write('\n')
            f.write(f'{datetime.datetime.now()}: Starting the shutdown...\n')
            f.write(f'Best model: Training {metrics_to_string(final_train_metrics)}\t'
                    f'VL:{metrics_to_string(final_val_metrics)}\n')
            f.write(200*'-')
            f.write('\n\n')
            f.write(200*'-')
            f.write('\n')
            f.write((f'{datetime.datetime.now()}: '
                     f'Shutdown completed\n'))
        return 'Done'
