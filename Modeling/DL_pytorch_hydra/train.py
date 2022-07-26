import numpy as np
import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
from srcs.trainer import Trainer
from srcs.utils import instantiate, get_logger


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_worker(config):
    """
    Train worker
    """
    logger = get_logger('train')
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader, is_func=False)
    # data_loader, valid_data_loader = instantiate(config.data_loader, is_func=True)

    # build model. print it's structure and # trainable params.
    print(config.arch)
    model = instantiate(config.arch, is_func=False)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(model)
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

def init_worker(rank, ngpus, working_dir, config):
    """
    Init worker
    """
    # initialize training config
    config = OmegaConf.create(config)
    config.local_rank = rank
    config.cwd = working_dir
    # print(config)
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # TODO: Timeout error: The client socket has timed out after 1800s while trying to connect to 127.0.0.1, 34567

    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='tcp://127.0.0.1:34567',
    #     world_size=ngpus,
    #     rank=rank)

    # Comment out in case of using only cpu

    # torch.cuda.set_device(rank)

    # start training processes
    print('Start train worker')
    train_worker(config)

@hydra.main(config_path='conf/', config_name='train')
def main(config):
    """
    Main function
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # assert n_gpu, 'Can\'t find any GPU device on this machine.'

    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))
    # print(working_dir)

    if config.resume is not None:
        config.resume = hydra.utils.to_absolute_path(config.resume)
    config = OmegaConf.to_yaml(config, resolve=True)
    # print(config)
    print(n_gpu)
    # torch.multiprocessing.spawn(init_worker, nprocs=n_gpu, args=(n_gpu, working_dir, config))
    init_worker(1, n_gpu, working_dir, config)
    print("Hello")

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()

