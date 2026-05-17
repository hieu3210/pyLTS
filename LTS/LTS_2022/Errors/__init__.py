# Forecasting errors assessment measurements
import math


class Measure:
    # Constructor with actual historical data
    def __init__(self, actual, forecasted):
        self.actual = actual
        self.forecasted = forecasted
        self.length = len(actual)

    # Error of forecasted results
    def error(self):
        err = []
        for i in range(self.length):
            err.append(abs(self.forecasted[i] - self.actual[i]))
        return err

    # Error of forecasted results (percentage)
    def error_percentage(self, num_digits):
        err = []
        for i in range(self.length):
            err.append(round(self.error()[i] / self.actual[i] * 100, num_digits))
        return err

    # Mean Absolute Error
    def mae(self):
        err = 0
        for i in range(self.length):
            err += self.error()[i]
        return err / self.length

    # Mean Squared Error
    def mse(self):
        err = 0
        for i in range(self.length):
            err += math.pow(self.error()[i], 2)
        return err / self.length

    # Root Mean Squared Error
    def rmse(self):
        return math.sqrt(self.mse())

    # Mean Absolute Percentage Error (%)
    def mape(self, num_digits):
        err = 0
        for i in range(self.length):
            err += abs((self.error()[i]) / self.actual[i])
        return round((err / self.length * 100), num_digits)

    # Symmetric Mean Absolute Percentage Error (%)
    def smape(self, num_digits):
        err = 0
        for i in range(self.length):
            err += abs(self.error()[i]) / ((abs(self.actual[i]) + abs(self.forecasted[i])) / 2)
        return round((err / self.length * 100), num_digits)

    # Mean Absolute Scaled Error (%)
    def mase(self):
        pass
