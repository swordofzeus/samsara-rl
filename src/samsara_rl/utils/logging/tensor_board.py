from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(f"{self.log_dir}/{self.experiment_name}")

    def log_metric(self, epoch: int, metric_name: str, value: float) -> None:
        self.writer.add_scalar(metric_name, value, global_step=epoch)

    def log_metrics(self, epoch: int, metric_name: str, values: dict[str, float]) -> None:
        self.writer.add_scalars(metric_name, values, global_step=epoch)

    def log_figure(self, epoch: int, tag: str, figure: Any) -> None:
        self.writer.add_figure(tag, figure, global_step=epoch)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def flush(self) -> None:
        self.writer.flush()
