import os
import pandas as pd
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from srcs.utils import get_logger


class TensorboardWriter():
    """
    Tensorboard:
    """
    def __init__(self, log_dir, enabled):
        self.logger = get_logger('tensorboard-writer')
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.timer = datetime.now()

    def set_step(self, step):
        """"""
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            attr = getattr(self.writer, name)
            return attr


class BatchMetrics:
    """
    """
    def __init__(self, *keys, postfix='', writer=None):
        """
        postfix is set either /train or /valid
        """
        self.writer = writer
        self.postfix = postfix
        if postfix:
            keys = [k + postfix for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """
        Set all values in dataframe equal to 0
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        """
        if self.postfix:
            key = key + self.postfix
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        """
        if self.postfix:
            key = key + self.postfix
        return self._data.average[key]

    def result(self):
        """
        """
        return dict(self._data.average)


class EpochMetrics:
    """
    Instances: logger, data (dataframe to store metrics), topk_idx (save best epoch idx)
    """
    def __init__(self, metric_names, phases=('train', 'valid'), monitoring='off'):
        """
        Parameters:
        - metric names: name of all the metrics
        """
        self.logger = get_logger('epoch-metrics')
        # setup pandas DataFrame with hierarchical columns
        # product function return cartesian product, which is all the possible combination
        columns = tuple(product(metric_names, phases))
        self._data = pd.DataFrame(columns=columns)  # TODO: add epoch duration
        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(monitoring)
        self.topk_idx = []

    def minimizing_metric(self, idx):
        """
        Get the metric of one epoch from data 
        If monitor_mode is min then set the same metric. If monitor_mode is max then add the subtraction before metric
        """
        if self.monitor_mode == 'off':
            return 0
        try:
            # Get metric as a row from the dataframe by access its index
            metric = self._data[self.monitor_metric].loc[idx]
        except KeyError:
            self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            return 0
        if self.monitor_mode == 'min':
            return metric
        else:
            return - metric

    def _parse_monitoring_mode(self, monitor_mode):
        """
        Parameters: monitor_mode consist of monitor_mode and monitor_metric. By default is set "off"
        """
        if monitor_mode == 'off':
            return 'off', None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ['min', 'max']
        return monitor_mode, monitor_metric

    def is_improved(self):
        """
        Get the index of last epoch and the index of best epoch and compare if they are the same
        If true: is improved; If false: no improved
        """
        if self.monitor_mode == 'off':
            return True

        last_epoch = self._data.index[-1]
        best_epoch = self.topk_idx[0]
        return last_epoch == best_epoch

    def keep_topk_checkpt(self, checkpt_dir, k=3):
        """
        Keep top-k checkpoints by deleting k+1'th best epoch index from dataframe for every epoch.
        """
        if len(self.topk_idx) > k and self.monitor_mode != 'off':
            last_epoch = self._data.index[-1]
            self.topk_idx = self.topk_idx[:(k + 1)]
            if last_epoch not in self.topk_idx:
                to_delete = last_epoch
            else:
                to_delete = self.topk_idx[-1]

            # delete checkpoint having out-of topk metric
            filename = str(checkpt_dir / 'checkpoint-epoch{}.pth'.format(to_delete.split('-')[1]))
            try:
                os.remove(filename)
            except FileNotFoundError:
                # this happens when current model is loaded from checkpoint
                # or target file is already removed somehow
                pass

    def update(self, epoch, result):
        """
        Update result to the metric dataframe with row name is the epoch index
        Save epoch index to topk_idx
        Sort the topk_idx according to the metrics
        """
        # print(type(result))
        # print(result.items())
        epoch_idx = f'epoch-{epoch}'
        self._data.loc[epoch_idx] = {tuple(k.split('/')): v for k, v in result.items()}

        self.topk_idx.append(epoch_idx)
        self.topk_idx = sorted(self.topk_idx, key=self.minimizing_metric)

    def latest(self):
        """
        Get the last element in the dataframe
        """
        return self._data[-1:]

    def to_csv(self, save_path=None):
        """
        Log metric data to csv file
        """
        self._data.to_csv(save_path)

    def __str__(self):
        """
        Return the string representation of metric data
        """
        return str(self._data)
