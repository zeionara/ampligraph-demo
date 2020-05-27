import logging
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import click
from ampligraph.datasets import load_from_csv
from ampligraph.utils import save_model, restore_model, create_tensorboard_visualizations

from utils.evaluation import compute_metrics, score_samples, summarize
from utils.timers import measure_execution_time
from utils.training import split, train_transe


@click.group()
def main():
    pass


@click.command()
@click.argument('train-file-path', type=str)  # Input csv file containing source graph
@click.argument('evaluation-file-path', type=str)  # Input csv file containing source graph
@click.argument('model-path', type=str, default='assets/models/transe.pkl')  # Where to store trained model
@click.argument('tensorboard-visualizations-path', type=str, default='./tensorboard-visualizations/game-of-thrones')
@measure_execution_time
def predict_missing_links(train_file_path, evaluation_file_path, model_path, tensorboard_visualizations_path):
    graph = load_from_csv('.', train_file_path, sep=',')
    evaluation_samples = load_from_csv('.', evaluation_file_path, sep=',')

    print('Head of the loaded graph: ')
    print(graph[:5])

    train_samples, test_samples = split(graph)
    print(f'Divided into train and test subsets with shapes {train_samples.shape} and {test_samples.shape} respectively.')

    if not os.path.isfile(model_path):
        model = train_transe(train_samples)  # train_complex(train_samples)
        save_model(model, model_path)
    else:
        model = restore_model(model_path)

    metrics = compute_metrics(model, train_samples, test_samples)
    print(f'{"metric":10s}: {"score":5s}')
    for metric, score in metrics.items():
        print(f'{metric:10s}: {score:<5.2f}')

    scores, ranks = score_samples(model, evaluation_samples, train_samples)
    evaluation_summary = summarize(scores, evaluation_samples, ranks)

    print(evaluation_summary)

    if tensorboard_visualizations_path:
        os.makedirs(tensorboard_visualizations_path, exist_ok=True)
        create_tensorboard_visualizations(model, tensorboard_visualizations_path)


if __name__ == '__main__':
    main.add_command(predict_missing_links)
    main()
