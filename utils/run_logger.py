import csv
import os
import time


class RunLogger:
    HEADER = [
        "frame", "t",
        "ego_x", "ego_y", "ego_yaw_rad", "ego_speed",
        "target_x", "target_y", "target_speed",
        "heading_error", "speed_error",
        "steer", "throttle", "brake",
        "current_idx",
    ]

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"run_{ts}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._frame = 0
        self._t0 = time.time()

    def log(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
        ego_speed: float,
        target_x: float,
        target_y: float,
        target_speed: float,
        heading_error: float,
        speed_error: float,
        steer: float,
        throttle: float,
        brake: float,
        current_idx: int,
    ) -> None:
        self._frame += 1
        self._writer.writerow([
            self._frame,
            f"{time.time() - self._t0:.3f}",
            f"{ego_x:.3f}",
            f"{ego_y:.3f}",
            f"{ego_yaw_rad:.4f}",
            f"{ego_speed:.3f}",
            f"{target_x:.3f}",
            f"{target_y:.3f}",
            f"{target_speed:.3f}",
            f"{heading_error:.4f}",
            f"{speed_error:.3f}",
            f"{steer:.4f}",
            f"{throttle:.4f}",
            f"{brake:.4f}",
            current_idx,
        ])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
