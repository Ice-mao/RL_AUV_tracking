from tianshou.trainer.base import BaseTrainer

class TwobufferTrainer(BaseTrainer):
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