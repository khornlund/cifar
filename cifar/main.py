import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import imgaug as ia

import cifar.data_loader.data_loaders as module_data
import cifar.model.loss as module_loss
import cifar.model.metric as module_metric
import cifar.model.model as module_arch
from cifar.trainer import Trainer
from cifar.utils import setup_logger, setup_logging


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Runner:

    def train(self, config, resume):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])

        self.logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config)
        valid_data_loader = data_loader.split_validation()

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)
        model, device = self._prepare_device(model, config['n_gpu'])

        self.logger.debug('Getting loss and metric function handles')
        loss = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        self.logger.debug('Initialising trainer')
        trainer = Trainer(model, loss, metrics, optimizer,
                        resume=resume,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

        trainer.train()
        self.logger.debug('Finished!')

    def predict(self, config, model_checkpoint):
            setup_logging(config)
            self.logger = setup_logger(self, config['testing']['verbose'])
            self._seed_everything(config['seed'])

            self.logger.info(f'Using config:\n{config}')

            self.logger.debug('Getting data_loader instance')
            data_loader = getattr(module_data, config['data_loader']['type'])(
                config['testing']['data_dir'],
                batch_size=config['testing']['batch_size'],
                shuffle=False,
                validation_split=0.0,
                train=False,
                num_workers=config['testing']['num_workers'],
                verbose=config['testing']['verbose']
            )

            self.logger.debug('Building model architecture')
            model = get_instance(module_arch, 'arch', config)

            self.logger.debug(f'Loading checkpoint {model_checkpoint}')
            checkpoint = torch.load(model_checkpoint)
            state_dict = checkpoint['state_dict']
            if config['n_gpu'] > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)

            # prepare model for testing
            model, device = self._prepare_device(model, config['n_gpu'])
            model.eval()

            all_preds = torch.zeros((len(data_loader.dataset), model.num_classes))
            self.logger.debug('Starting...')
            with torch.no_grad():
                for i, data in enumerate(tqdm(data_loader)):
                    data = data.to(device)
                    output = model(data)
                    batch_size = output.shape[0]
                    batch_preds = output.max(1)[1]  # argmax
                    all_preds[i * batch_size:(i + 1) * batch_size, :] = batch_preds.cpu()

            # wrangle and save predictions

            raw_df = pd.DataFrame(preds.numpy())

            # do something with predictions

            predictions_filename = os.path.join(
                config['testing']['data_dir'],
                config['name'] + '_preds.csv')
            raw_df.to_csv(predictions_filename)
            self.logger.info(f'Finished saving predictions to "{predictions_filename}"')

    def _prepare_device(self, model, n_gpu_use):
        device, device_ids = self._get_device(n_gpu_use)
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model, device

    def _get_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                                f"but only {n_gpu} are available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        self.logger.info(f'Using device: {device}, {list_ids}')
        return device, list_ids

    def _seed_everything(self, seed):
        self.logger.info(f'Using random seed: {seed}')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        ia.seed(seed)
