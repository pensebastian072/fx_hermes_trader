"""Broker interface. The only implementation in Milestone 1 is the paper broker."""

from abc import ABC, abstractmethod

from app.schemas.trades import PaperOrder


class BrokerBase(ABC):
    @abstractmethod
    def create_order(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        target: float,
        size: float,
        risk_percent: float,
    ) -> PaperOrder: ...

    @abstractmethod
    def close_position(self, order_id: str, price: float) -> float: ...

    @abstractmethod
    def open_positions(self) -> list[PaperOrder]: ...
