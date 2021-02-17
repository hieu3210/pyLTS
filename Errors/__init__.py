# Forecasting errors assessment measurements
import math


class Measure:
    # Constructor with actual historical data
    def __init__(self, actual, forecasted):
        self.__actual = actual
        self.__forecasted = forecasted
        self.__length = len(actual)

    # Error of forecasted results
    def error(self):
        err = []
        for i in range(self.__length):
            err.append(abs(self.__forecasted[i] - self.__actual[i]))
        return err

    # Error of forecasted results (percentage)
    def error_percentage(self, round_limit):
        err = []
        for i in range(self.__length):
            err.append(round(self.error()[i] / self.__actual[i] * 100, round_limit))
        return err

    # Mean Squared Error
    def mse(self):
        err = 0
        for i in range(self.__length):
            err += math.pow(self.error()[i], 2)
        return err / self.__length

    # Root Mean Squared Error
    def rmse(self):
        return math.sqrt(self.mse())

    # Mean Absolute Percentage Error (%)
    def mape(self, round_limit):
        err = 0
        for i in range(self.__length):
            err += abs((self.error()[i]) / self.__actual[i])
        return round((err / self.__length * 100), round_limit)
