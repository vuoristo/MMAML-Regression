import os
import json
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

from maml.datasets.simple_functions import (
    SinusoidMetaDataset,
    LinearMetaDataset,
    MixedFunctionsMetaDataset,
    ManyFunctionsMetaDataset,
    FiveFunctionsMetaDataset,
    MultiSinusoidsMetaDataset,
)
from maml.models.fully_connected import FullyConnectedModel, MultiFullyConnectedModel
from maml.models.gated_net import GatedNet
from maml.models.lstm_embedding_model import LSTMEmbeddingModel
from maml.metalearner import MetaLearner
from maml.trainer import Trainer
from maml.utils import optimizer_to_device, get_git_revision_hash


def main(args):
    is_training = not args.eval
    run_name = 'train' if is_training else 'eval'

    if is_training:
        writer = SummaryWriter('./train_dir/{0}/{1}'.format(
            args.output_folder, run_name))
    else:
        writer = None

    save_folder = './train_dir/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    config_name = '{0}_config.json'.format(run_name)
    with open(os.path.join(save_folder, config_name), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        config.update({'git_hash': get_git_revision_hash()})
        json.dump(config, f, indent=2)

    _num_tasks = 1
    if args.dataset == 'sinusoid':
        dataset = SinusoidMetaDataset(
            num_total_batches=args.num_batches,
            num_samples_per_function=args.num_samples_per_class,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            amp_range=args.amp_range,
            phase_range=args.phase_range,
            input_range=args.input_range,
            oracle=args.oracle,
            train=is_training,
            device=args.device)
        loss_func = torch.nn.MSELoss()
        collect_accuracies = False
    elif args.dataset == 'linear':
        dataset = LinearMetaDataset(
            num_total_batches=args.num_batches,
            num_samples_per_function=args.num_samples_per_class,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            slope_range=args.slope_range,
            intersect_range=args.intersect_range,
            input_range=args.input_range,
            oracle=args.oracle,
            train=is_training,
            device=args.device)
        loss_func = torch.nn.MSELoss()
        collect_accuracies = False
    elif args.dataset == 'mixed':
        dataset = MixedFunctionsMetaDataset(
            num_total_batches=args.num_batches,
            num_samples_per_function=args.num_samples_per_class,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            amp_range=args.amp_range,
            phase_range=args.phase_range,
            slope_range=args.slope_range,
            intersect_range=args.intersect_range,
            input_range=args.input_range,
            noise_std=args.noise_std,
            oracle=args.oracle,
            task_oracle=args.task_oracle,
            train=is_training,
            device=args.device)
        loss_func = torch.nn.MSELoss()
        collect_accuracies = False
        _num_tasks = 2
    elif args.dataset == 'five':
        dataset = FiveFunctionsMetaDataset(
            num_total_batches=args.num_batches,
            num_samples_per_function=args.num_samples_per_class,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            amp_range=args.amp_range,
            phase_range=args.phase_range,
            slope_range=args.slope_range,
            intersect_range=args.intersect_range,
            input_range=args.input_range,
            noise_std=args.noise_std,
            oracle=args.oracle,
            task_oracle=args.task_oracle,
            train=is_training,
            device=args.device)
        loss_func = torch.nn.MSELoss()
        collect_accuracies = False
        _num_tasks = 5
    elif args.dataset == 'many':
        dataset = ManyFunctionsMetaDataset(
            num_total_batches=args.num_batches,
            num_samples_per_function=args.num_samples_per_class,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            amp_range=args.amp_range,
            phase_range=args.phase_range,
            slope_range=args.slope_range,
            intersect_range=args.intersect_range,
            input_range=args.input_range,
            noise_std=args.noise_std,
            oracle=args.oracle,
            task_oracle=args.task_oracle,
            train=is_training,
            device=args.device)
        loss_func = torch.nn.MSELoss()
        collect_accuracies = False
        _num_tasks = 3
    elif args.dataset == 'multisinusoids':
        dataset = MultiSinusoidsMetaDataset(
            num_total_batches=args.num_batches,
            num_samples_per_function=args.num_samples_per_class,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            amp_range=args.amp_range,
            phase_range=args.phase_range,
            slope_range=args.slope_range,
            intersect_range=args.intersect_range,
            input_range=args.input_range,
            noise_std=args.noise_std,
            oracle=args.oracle,
            task_oracle=args.task_oracle,
            train=is_training,
            device=args.device)
        loss_func = torch.nn.MSELoss()
        collect_accuracies = False
    else:
        raise ValueError('Unrecognized dataset {}'.format(args.dataset))

    embedding_model = None

    if args.model_type == 'fc':
        model = FullyConnectedModel(
            input_size=np.prod(dataset.input_size),
            output_size=dataset.output_size,
            hidden_sizes=args.hidden_sizes,
            disable_norm=args.disable_norm,
            bias_transformation_size=args.bias_transformation_size)
    elif args.model_type == 'multi':
        model = MultiFullyConnectedModel(
            input_size=np.prod(dataset.input_size),
            output_size=dataset.output_size,
            hidden_sizes=args.hidden_sizes,
            disable_norm=args.disable_norm,
            num_tasks=_num_tasks,
            bias_transformation_size=args.bias_transformation_size)
    elif args.model_type == 'gated':
        model = GatedNet(
            input_size=np.prod(dataset.input_size),
            output_size=dataset.output_size,
            hidden_sizes=args.hidden_sizes,
            condition_type=args.condition_type,
            condition_order=args.condition_order)
    else:
        raise ValueError('Unrecognized model type {}'.format(args.model_type))
    model_parameters = list(model.parameters())

    if args.embedding_type == '':
        embedding_model = None
    elif args.embedding_type == 'LSTM':
        embedding_model = LSTMEmbeddingModel(
            input_size=np.prod(dataset.input_size),
            output_size=dataset.output_size,
            embedding_dims=args.embedding_dims,
            hidden_size=args.embedding_hidden_size,
            num_layers=args.embedding_num_layers)
        embedding_parameters = list(embedding_model.parameters())
    else:
        raise ValueError('Unrecognized embedding type {}'.format(
            args.embedding_type))

    optimizers = None
    if embedding_model:
        optimizers = (torch.optim.Adam(model_parameters, lr=args.slow_lr),
                      torch.optim.Adam(embedding_parameters, lr=args.slow_lr))
    else:
        optimizers = (torch.optim.Adam(model_parameters, lr=args.slow_lr), )

    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        if 'optimizer' in checkpoint:
            pass
        else:
            optimizers[0].load_state_dict(checkpoint['optimizers'][0])
            optimizer_to_device(optimizers[0], args.device)

            if embedding_model:
                embedding_model.load_state_dict(
                    checkpoint['embedding_model_state_dict'])
                optimizers[1].load_state_dict(checkpoint['optimizers'][1])
                optimizer_to_device(optimizers[1], args.device)

    meta_learner = MetaLearner(
        model, embedding_model, optimizers, fast_lr=args.fast_lr,
        loss_func=loss_func, first_order=args.first_order,
        num_updates=args.num_updates,
        inner_loop_grad_clip=args.inner_loop_grad_clip,
        collect_accuracies=collect_accuracies, device=args.device,
        embedding_grad_clip=args.embedding_grad_clip,
        model_grad_clip=args.model_grad_clip)

    trainer = Trainer(
        meta_learner=meta_learner, meta_dataset=dataset, writer=writer,
        log_interval=args.log_interval, save_interval=args.save_interval,
        model_type=args.model_type, save_folder=save_folder)

    if is_training:
        trainer.train()
    else:
        trainer.eval()


if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Multimodal Model-Agnostic Meta-Learning (MAML)')

    # Model
    parser.add_argument('--hidden-sizes', type=int,
                        default=[256, 128, 64, 64], nargs='+',
                        help='number of hidden units per layer')
    parser.add_argument('--model-type', type=str, default='fc',
                        help='type of the model')
    parser.add_argument('--condition-type', type=str, default='affine',
                        help='type of the conditional layers')
    parser.add_argument('--use-max-pool', action='store_true',
                        help='choose whether to use max pooling with convolutional model')
    parser.add_argument('--num-channels', type=int, default=64,
                        help='number of channels in convolutional layers')
    parser.add_argument('--disable-norm', action='store_true',
                        help='disable batchnorm after linear layers in a fully connected model')
    parser.add_argument('--bias-transformation-size', type=int, default=0,
                        help='size of bias transformation vector that is concatenated with '
                        'input')
    parser.add_argument('--condition-order', type=str, default='low2high',
                        help='order of the conditional layers to be used')

    # Embedding
    parser.add_argument('--embedding-type', type=str, default='',
                        help='type of the embedding')
    parser.add_argument('--embedding-hidden-size', type=int, default=40,
                        help='number of hidden units per layer in recurrent embedding model')
    parser.add_argument('--embedding-num-layers', type=int, default=2,
                        help='number of layers in recurrent embedding model')
    parser.add_argument('--embedding-dims', type=int, nargs='+', default=0,
                        help='dimensions of the embeddings')

    # Randomly sampled embedding vectors
    parser.add_argument('--num-sample-embedding', type=int, default=0,
                        help='number of randomly sampled embedding vectors')
    parser.add_argument(
        '--sample-embedding-file', type=str, default='embeddings',
        help='the file name of randomly sampled embedding vectors')
    parser.add_argument(
        '--sample-embedding-file-type', type=str, default='hdf5')

    # Inner loop
    parser.add_argument('--first-order', action='store_true',
                        help='use the first-order approximation of MAML')
    parser.add_argument('--fast-lr', type=float, default=0.4,
                        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--inner-loop-grad-clip', type=float, default=0.0,
                        help='enable gradient clipping in the inner loop')
    parser.add_argument('--num-updates', type=int, default=1,
                        help='how many update steps in the inner loop')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=1920000,
                        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=32,
                        help='number of tasks per batch')
    parser.add_argument('--slow-lr', type=float, default=0.001,
                        help='learning rate for the global update of MAML')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
                        help='name of the output folder')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='how many DataLoader workers to use')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='number of batches between tensorboard writes')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='number of batches between model saves')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to saved parameters.')

    # Dataset
    parser.add_argument('--dataset', type=str, default='omniglot',
                        help='which dataset to use')
    parser.add_argument('--data-root', type=str, default='data',
                        help='path to store datasets')
    parser.add_argument('--num-samples-per-class', type=int, default=1,
                        help='how many samples per class for training')
    parser.add_argument('--num-val-samples', type=int, default=1,
                        help='how many samples per class for validation')
    parser.add_argument('--input-range', type=float, default=[-5.0, 5.0],
                        nargs='+', help='input range of simple functions')
    parser.add_argument('--phase-range', type=float, default=[0, np.pi],
                        nargs='+', help='phase range of sinusoids')
    parser.add_argument('--amp-range', type=float, default=[0.1, 5.0],
                        nargs='+', help='amp range of sinusoids')
    parser.add_argument('--slope-range', type=float, default=[-3.0, 3.0],
                        nargs='+', help='slope range of linear functions')
    parser.add_argument('--intersect-range', type=float, default=[-3.0, 3.0],
                        nargs='+', help='intersect range of linear functions')
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help='add gaussian noise to mixed functions')
    parser.add_argument('--oracle', action='store_true',
                        help='concatenate phase and amp to sinusoid inputs')
    parser.add_argument('--task-oracle', action='store_true',
                        help='uses task id for prediction in some models')

    parser.add_argument('--embedding-grad-clip', type=float, default=2.0,
                        help='')
    parser.add_argument('--model-grad-clip', type=float, default=2.0,
                        help='')

    args = parser.parse_args()

    if args.embedding_dims == 0:
        args.embedding_dims = args.hidden_sizes

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # Make sure num sample embedding < num sample tasks
    args.num_sample_embedding = min(
        args.num_sample_embedding, args.num_batches)

    # Device
    args.device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    main(args)
