from grpc import server
import torch
from sklearn.model_selection import StratifiedKFold
from utils.messages import CV_ValidationMessage, ClientMessage, HoldOut_ValidationMessage
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import pandas as pd
import os

from utils.util import create_model_name, create_model_name_state


class RemoteLocalClient:
    """
    ## Description:

    Prototype of a generic Local Client. 
    """

    state_best_f1: float

    def __init__(self, config):

        self.id = config.metadata.id
        self.ckpt_epoch = config.metadata.ckpt_epoch
        self.ckpt_best = config.metadata.ckpt_best
        self.log_path = config.metadata.log_path
        self.metrics = config.metadata.metrics
        self.train_history = {'loss': []}
        self.val_history = {'loss': []}
        self.personalized_train_history = {'loss': []}
        self.personalized_val_history = {'loss': []}
        for key in self.metrics.keys():
            self.train_history[key] = []
            self.val_history[key] = []
            self.personalized_train_history[key] = []
            self.personalized_val_history[key] = []
        self.dataset_path = config.metadata.dev_path
        self.test_set_path = config.metadata.test_path
        self.train_path = config.metadata.train_path
        self.val_path = config.metadata.val_path
        self.train_path_orig = config.metadata.train_path_orig
        self.val_path_orig = config.metadata.val_path_orig

        self.global_iter_id = 0
        self.state_id = 0

        self.save_all_models = False
        self.save_state_models = False

        self.save_all_models_path = config.metadata.save_all_models_path
        self.save_state_models_path = config.metadata.save_state_models_path

        self.history_path = f'{os.path.dirname(self.log_path)}/../History'
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

        self.n_classes = config.metadata.n_classes

        self.model = None
        self.optimizer = None

        self.fold_split = []
        self.current_iteration = 0
        self.state_best_f1 = -float('inf')

        log_dir = os.path.dirname(self.log_path)
        ckpt_best_dir = os.path.dirname(self.ckpt_best)
        ckpt_epoch_dir = os.path.dirname(self.ckpt_epoch)
        if not (log_dir == '') and not (os.path.exists(log_dir)):
            os.makedirs(log_dir)
        if not (ckpt_best_dir == '') and not (os.path.exists(ckpt_best_dir)):
            os.makedirs(ckpt_best_dir)
        if not (ckpt_epoch_dir == '') and not (os.path.exists(ckpt_epoch_dir)):
            os.makedirs(ckpt_epoch_dir)

        self.config = config

    def activate(self, mode):
        with open(self.log_path, mode) as f:
            f.write(
                f'{datetime.datetime.now()}: '
                f'Local Client {self.id}: Attivato\n'
                f'Client Metadata :\n{self.config.metadata}\n'
                f'Client Training Params :\n{self.config.training_params}\n\n')

    def get_id(self):
        return self.id

    def _compute_class_support(self, dataset, class_id, target_label):
        """
        It computes the number of records labelled with `class_id`

        Args:
            `dataset`: (Dataframe)
                The target dataset

            `class_id`: (str)
                The name of the class to look for

            `target_label`: (str)
                The name of the dataset attribute which represents the target label

        Returns:
            int: 
                The number of instances labelled as `class_id` in the dataset.
        """
        df_proj = dataset[dataset[target_label] == class_id]
        return len(df_proj)

    def _compute_support(self, dataset, target_label):
        """
        It computes the support for each target class in the dataset

        Args:

        `dataset`: (Dataframe)
            The target dataset

        `target_label`: (str)
            The name of the dataset attribute which represents the target label 

        Returns:
            dict: 
                A dictionary of the form `{key : support}`,
                which contains the number of records labelled with `key` in the given dataset

        """
        support_dict = {}
        for class_id in range(self.n_classes):
            support_dict[str(class_id)] = self._compute_class_support(
                dataset, class_id, target_label)
        return support_dict

    def _split_dataset(self, dataset, validation_message, target_label):
        """
        It splits the given dataset into train and validation, according to the strategy specified by the `validation_message`

        As:

        `dataset`: (Dataframe)
            The dataset to split

        `validation_message` : (CV_ValidationMessage or HoldOut_ValidationMessage)
            The strategy to follow when splitting the dataset.

        `target_label` : (str)
            The name of the dataset attribute which represents the target label 

        Returns:
            None:
                The result is saved into the `self.fold_split` field.
                It is a list made up of pairs of the form (train,val).
                The i-th entry stands for the training set and the validation set of the i-th fold.
                NOTE: If the Hold-Out strategy was applied, then the list has just one entry.
        """

        columns = [col for col in dataset.columns if col != target_label]

        if isinstance(validation_message, CV_ValidationMessage):
            n_folds = validation_message.total_folds
            kfold = StratifiedKFold(n_folds, shuffle=True)

            for train, val in kfold.split(dataset[columns].values,
                                          dataset[target_label].values):
                self.fold_split.append((train, val))

        elif isinstance(validation_message, HoldOut_ValidationMessage):
            val_perc = validation_message.val_split_percentage
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=val_perc, random_state=42)

            for train, val in sss.split(dataset[columns].values,
                                        dataset[target_label].values):
                self.fold_split.append((train, val))
            assert len(self.fold_split) == 1

        else:
            raise KeyError(
                'The validation strategy must be one of CV_ValidationMessage' +
                ' or HoldOut_ValidationMessage')

    def _compute_class_weights(self, dataset, target_label):
        """
        It computes the weight for each possible target class in the dataset.
        The weight of class c, for a dataset made up of N records and with C different classes, is computed as follows:
        $$
        w_{c} = \\frac{N}{S(c) \\times C}
        $$
        where S(c) is the number of records labelled as c.

        Returns:
        Tensor:
            Tensor of length C, containing the weights for all the different classes.
        NOTE:
            This function is applied just to compute the class weights that have to be given in input to the local loss function.
        """
        weights = []
        n_samples = len(dataset)
        n_classes = self.n_classes
        # print('N classes: ', n_classes)
        for class_id in range(n_classes):
            n_samples_class = len(
                dataset[dataset[target_label] == class_id])
            weight_class = 0
            if n_samples_class > 0:
                weight_class = round(n_samples/(n_classes*n_samples_class), 3)
            weights.append(weight_class)

        weights = torch.Tensor(weights)
        return weights

    def _evaluate_model(self, data_loader, compute_loss=False):
        """
        ## Description:

        It evaluates the current model w.r.t. the chosen metrics.
        """
        raise NotImplementedError(
            'This method must be implemented in the subclass')

    def get_weights(self, validation_message):
        """
        ## Description:

        It computes the weights associated to the local clients, i.e. the number of records contained in the training and in the validation set.

        ## Args:

        `validation_message`: (ValidationMessage)
            The message received from the parent node

        ## Returns:

        ### \( (w_{t},w_{v}) : (int, int) \)

        A tuple containing the weight associated to the training and to the validation set, respectively.
        """
        if validation_message.from_file:
            train_set = pd.read_csv(self.train_path)
            val_set = pd.read_csv(self.val_path)

        else:
            if len(self.fold_split) == 0:
                dataset = pd.read_csv(self.dataset_path)
                self._split_dataset(dataset, validation_message)
                del dataset

            if isinstance(validation_message, CV_ValidationMessage):
                current_fold = validation_message.current_fold
                train_set = self.fold_split[current_fold][0]
                val_set = self.fold_split[current_fold][1]
                with open(self.log_path, 'a') as f:
                    f.write(
                        f'{datetime.datetime.now()}: '
                        f'KCV: Eseguo la validazione sul fold '
                        f'{current_fold+1}/{len(self.fold_split)}\n')

            else:
                train_set = self.fold_split[0][0]
                val_set = self.fold_split[0][1]

        train_weight = len(train_set)
        val_weight = len(val_set)
        with open(self.log_path, 'a') as f:
            f.write((f'{datetime.datetime.now()}: Get_Weights: '
                     f'Training: {train_weight} '
                     f'Validation: {val_weight}\n'))
        return train_weight, val_weight

    def personalize(self, server_message):
        raise NotImplementedError('This method is abstract')

    def shutdown(self):
        """
        ## Description:

        It shuts down the local client
        """
        with open(self.log_path, 'a') as f:
            f.write((f'{datetime.datetime.now()}: '
                     f'Shutdown completed\n'))
        return 'Done'

    def broadcast_train_msg(self, server_message):
        """
        ## Description:

        It performs the local training of the received model and eventually updates the internal state of the local client.
        ## Args:

        `server_message`: (ServerMessage)
        The message received from the parent node.

        ## Returns:

        ### [message: ClientMessage]

        A list with just one element. The message which is transmitted to the parent node, containing the updated model.

        NOTE:
            The method has to be implemented in the sub-class!
        """
        raise NotImplementedError(
            'This method must be implemented in the subclass')

    def evaluate_metrics(self, server_message):
        """
        ## Description:

        It performs the evaluation of the received model according to the predefined metrics. It eventually updates the internal state of the local client.

        ## Args:

        `server_message`: (ServerMessage)
        The message received from the parent node.

        ## Returns:

        ### [message: ClientMessage]

        A list with just one element. It contains the message which is transmitted to the parent node, containing the model evaluation.

        NOTE:
            The method has to be implemented in the sub-class!
        """
        raise NotImplementedError(
            'This method must be implemented in the subclass')
