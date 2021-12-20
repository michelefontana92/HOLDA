import numpy as np


def evaluate_metrics(metrics, y_true, y_pred):
    metrics_result = {}

    for metric, metric_fn in metrics.items():
        metrics_result[metric] = np.round(
            metric_fn(y_true, y_pred), 4)
    return metrics_result


def print_metrics(metrics_result):
    for metric, value in metrics_result.items():
        print(f'{metric}: {value}\t')


def metrics_to_string(metrics):
    return ''.join([f'{metric} : {value}\t'
                    for metric, value in metrics.items()])
