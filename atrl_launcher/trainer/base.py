import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import tqdm

from tianshou.data import (
    AsyncCollector,
    CollectStats,
    EpochStats,
    InfoStats,
    ReplayBuffer,
    SequenceSummaryStats,
)
from tianshou.data.collector import BaseCollector, CollectStatsBase
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    MovAvg,
    tqdm_config,
)
from tianshou.utils.logging import set_numerical_fields_to_precision
from tianshou.utils.torch_utils import policy_within_training_step
from tianshou.trainer.base import BaseTrainer


class SaveOnEpochTrainer(BaseTrainer):
    """
     base trainer for save policy every epoch
     new param is save_epoch_fn
    """
    def __init__(
            self,
            policy: BasePolicy,
            max_epoch: int,
            batch_size: int | None,
            train_collector: BaseCollector | None = None,
            test_collector: BaseCollector | None = None,
            buffer: ReplayBuffer | None = None,
            step_per_epoch: int | None = None,
            repeat_per_collect: int | None = None,
            episode_per_test: int | None = None,
            update_per_step: float = 1.0,
            step_per_collect: int | None = None,
            episode_per_collect: int | None = None,
            train_fn: Callable[[int, int], None] | None = None,
            test_fn: Callable[[int, int | None], None] | None = None,
            stop_fn: Callable[[float], bool] | None = None,
            save_best_fn: Callable[[BasePolicy], None] | None = None,
            save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
            resume_from_log: bool = False,
            reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
            logger: BaseLogger = LazyLogger(),
            verbose: bool = True,
            show_progress: bool = True,
            test_in_train: bool = True,
            # new part
            save_epoch_fn: Callable[[BasePolicy, int], None] | None = None,
    ):
        super().__init__(
            policy=policy,
            max_epoch=max_epoch,
            batch_size=batch_size,
            train_collector=train_collector,
            test_collector=test_collector,
            buffer=buffer,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=episode_per_test,
            update_per_step=update_per_step,
            step_per_collect=step_per_collect,
            episode_per_collect=episode_per_collect,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
        )
        self.save_epoch_fn = save_epoch_fn

    def __next__(self) -> EpochStats:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        print(self.epoch)
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        progress = tqdm.tqdm if self.show_progress else DummyTqdm

        # perform n step_per_epoch
        with progress(total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config) as t:
            train_stat: CollectStatsBase
            while t.n < t.total and not self.stop_fn_flag:
                train_stat, update_stat, self.stop_fn_flag = self.training_step()

                if isinstance(train_stat, CollectStats):
                    pbar_data_dict = {
                        "env_step": str(self.env_step),
                        "rew": f"{self.last_rew:.2f}",
                        "len": str(int(self.last_len)),
                        "n/ep": str(train_stat.n_collected_episodes),
                        "n/st": str(train_stat.n_collected_steps),
                    }
                    t.update(train_stat.n_collected_steps)
                else:
                    pbar_data_dict = {}
                    t.update()

                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["gradient_step"] = str(self._gradient_step)
                t.set_postfix(**pbar_data_dict)

                if self.stop_fn_flag:
                    break

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        # for epoch polict save
        if self.save_epoch_fn:
            self.save_epoch_fn(self.policy, self.epoch)

        # for offline RL
        if self.train_collector is None:
            assert self.buffer is not None
            batch_size = self.batch_size or len(self.buffer)
            self.env_step = self._gradient_step * batch_size

        test_stat = None
        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch,
                self.env_step,
                self._gradient_step,
                self.save_checkpoint_fn,
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()

        info_stat = gather_info(
            start_time=self.start_time,
            policy_update_time=self.policy_update_time,
            gradient_step=self._gradient_step,
            best_reward=self.best_reward,
            best_reward_std=self.best_reward_std,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
        )

        self.logger.log_info_data(asdict(info_stat), self.epoch)

        # in case trainer is used with run(), epoch_stat will not be returned
        return EpochStats(
            epoch=self.epoch,
            train_collect_stat=train_stat,
            test_collect_stat=test_stat,
            training_stat=update_stat,
            info_stat=info_stat,
        )

    def policy_update_fn(self, collect_stats: CollectStatsBase) -> TrainingStats:
        pass


class OffpolicyTrainer(SaveOnEpochTrainer):
    """Offpolicy trainer, samples mini-batches from buffer and passes them to update.

    Note that with this trainer, it is expected that the policy's `learn` method
    does not perform additional mini-batching but just updates params from the received
    mini-batch.
    """

    # for mypy
    assert isinstance(BaseTrainer.__doc__, str)
    __doc__ += BaseTrainer.gen_doc("offpolicy") + "\n".join(BaseTrainer.__doc__.split("\n")[1:])

    def policy_update_fn(
        self,
        # TODO: this is the only implementation where collect_stats is actually needed. Maybe change interface?
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Perform `update_per_step * n_collected_steps` gradient steps by sampling mini-batches from the buffer.

        :param collect_stats: the :class:`~TrainingStats` instance returned by the last gradient step. Some values
            in it will be replaced by their moving averages.
        """
        assert self.train_collector is not None
        n_collected_steps = collect_stats.n_collected_steps
        n_gradient_steps = round(self.update_per_step * n_collected_steps)
        if n_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={n_collected_steps}, "
                f"update_per_step={self.update_per_step}",
            )
        for _ in range(n_gradient_steps):
            update_stat = self._sample_and_update(self.train_collector.buffer)

            # logging
            self.policy_update_time += update_stat.train_time
        # TODO: only the last update_stat is returned, should be improved
        return update_stat


class OnpolicyTrainer(SaveOnEpochTrainer):
    """On-policy trainer, passes the entire buffer to .update and resets it after.

    Note that it is expected that the learn method of a policy will perform
    batching when using this trainer.
    """

    # for mypy
    assert isinstance(BaseTrainer.__doc__, str)
    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(BaseTrainer.__doc__.split("\n")[1:])

    def policy_update_fn(
        self,
        result: CollectStatsBase | None = None,
    ) -> TrainingStats:
        """Perform one on-policy update by passing the entire buffer to the policy's update method."""
        assert self.train_collector is not None
        training_stat = self.policy.update(
            sample_size=0,
            buffer=self.train_collector.buffer,
            # Note: sample_size is None, so the whole buffer is used for the update.
            # The kwargs are in the end passed to the .learn method, which uses
            # batch_size to iterate through the buffer in mini-batches
            # Off-policy algos typically don't use the batch_size kwarg at all
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )

        # just for logging, no functional role
        self.policy_update_time += training_stat.train_time
        # TODO: remove the gradient step counting in trainers? Doesn't seem like
        #   it's important and it adds complexity
        self._gradient_step += 1
        if self.batch_size is None:
            self._gradient_step += 1
        elif self.batch_size > 0:
            self._gradient_step += int((len(self.train_collector.buffer) - 0.1) // self.batch_size)

        # Note: this is the main difference to the off-policy trainer!
        # The second difference is that batches of data are sampled without replacement
        # during training, whereas in off-policy or offline training, the batches are
        # sampled with replacement (and potentially custom prioritization).
        self.train_collector.reset_buffer(keep_statistics=True)

        # The step is the number of mini-batches used for the update, so essentially
        self._update_moving_avg_stats_and_log_update_data(training_stat)

        return training_stat
