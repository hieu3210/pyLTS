# Forecasting errors assessment measurements
import math

class Measure:
    # Constructor with actual historical data
    def __init__(self, actual, forecasted):
        self.__actual = actual
        self.__forecasted = forecasted
        self.__length = len(actual)

    # Mean Squared Error
    def mse(self):
        err = 0
        for i in range(self.__length):
            err += math.pow(self.__actual[i] - self.__forecasted[i], 2)
        return err / self.__length

    # Root Mean Squared Error
    def rmse(self):
        return math.sqrt(self.mse())

    # MAPE (%)
    def mape(self):
        err = 0
        for i in range(self.__length):
            err += abs((self.__actual[i] - self.__forecasted[i]) / self.__actual[i])
        return err / self.__length * 100
