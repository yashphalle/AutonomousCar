class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float = 0.05,
        integral_limit: float = 2.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, error: float) -> float:
        self.integral += error * self.dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit

        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self) -> None:
        self.prev_error = 0.0
        self.integral = 0.0
