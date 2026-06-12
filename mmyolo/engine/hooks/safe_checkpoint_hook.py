# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import CheckpointHook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class SafeCheckpointHook(CheckpointHook):
    """Checkpoint hook tolerant of stale best-checkpoint metadata.

    A checkpoint resumed into a different work directory can retain the old
    best-checkpoint path in its message hub. Some MMEngine versions try to
    remove that missing path as a directory when the metric improves, raising
    ``FileNotFoundError`` after validation. Discarding only the stale path lets
    the parent hook save the new best checkpoint normally.
    """

    def _save_best_checkpoint(self, runner: Runner, metrics: dict) -> None:
        best_path = getattr(self, 'best_ckpt_path', None)
        if isinstance(best_path, str) and best_path:
            try:
                path_exists = self.file_backend.exists(best_path)
            except (AttributeError, NotImplementedError):
                path_exists = (self.file_backend.isfile(best_path)
                               or self.file_backend.isdir(best_path))

            if not path_exists:
                runner.logger.warning(
                    'Discarding stale best checkpoint path from resumed '
                    f'metadata: {best_path}')
                self.best_ckpt_path = None

        super()._save_best_checkpoint(runner, metrics)
