import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._step_train = 0
        self._step_val = 0
        self._step_test = 0

    def forward(self, x_in):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        raise NotImplementedError

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_train += 1  # =self.global_step
        return self._step(batch, batch_idx, "train", self._step_train, optimizer_idx)

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val, optimizer_idx)

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test, optimizer_idx)

    def _epoch_end(self, outputs: list, state: str):
        return

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")
