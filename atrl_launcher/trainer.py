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

class TwobufferTrainer(BaseTrainer):
    """
     inherit from BaseTrainer and OffpolicyTrainer to get training Batch
     from the demo buffer and reinforcement replay buffer equally
    """
    def __init__(
        self,
        policy: BasePolicy,
        max_epoch: int,
        batch_size: int | None,
        train_collector: BaseCollector | None = None,
        test_collector: BaseCollector | None = None,
        buffer: ReplayBuffer | None = None,
        demo_buffer: ReplayBuffer | None = None, # use for the TwobufferTrainer
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

        # 处理 demo_buffer
        self.demo_buffer = demo_buffer

    def policy_update_fn(
        self,
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
            update_stat = self._sample_and_update(self.train_collector.buffer, self.demo_buffer)

            # logging
            self.policy_update_time += update_stat.train_time
        # TODO: only the last update_stat is returned, should be improved
        return update_stat

    def _sample_and_update(self, buffer: ReplayBuffer, demo_buffer: ReplayBuffer) -> TrainingStats:
        """Sample a mini-batch, perform one gradient step, and update the _gradient_step counter."""
        self._gradient_step += 1
        # Note: since sample_size=batch_size, this will perform
        # exactly one gradient step. This is why we don't need to calculate the
        # number of gradient steps, like in the on-policy case.
        update_stat = self.policy.update(sample_size=self.batch_size, buffer1=buffer, buffer2=demo_buffer)
        self._update_moving_avg_stats_and_log_update_data(update_stat)
        return update_stat