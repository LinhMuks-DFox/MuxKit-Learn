import abc


class ExampleRunnable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self) -> None:
        pass
