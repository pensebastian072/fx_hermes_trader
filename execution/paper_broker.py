"""Paper broker simulator.

Simulates market orders with immediate fills at the requested entry price.
All activity is persisted append-only to orders.jsonl and fills.jsonl, and
the broker rebuilds its in-memory state from those logs on startup, so the
JSONL files are the source of truth.

There is no network access and no real broker in this module.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.schemas.trades import Fill, PaperOrder
from app.services.artifact_service import append_jsonl, read_jsonl
from execution.broker_base import BrokerBase


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperBroker(BrokerBase):
    def __init__(self, logs_dir: Path, starting_capital: float = 10000.0):
        self.orders_path = logs_dir / "orders.jsonl"
        self.fills_path = logs_dir / "fills.jsonl"
        self.starting_capital = float(starting_capital)
        self.equity = float(starting_capital)
        self._orders: dict[str, PaperOrder] = {}
        self._fills: list[Fill] = []
        self._replay_logs()

    def _replay_logs(self) -> None:
        for rec in read_jsonl(self.orders_path):
            order = PaperOrder(**rec)
            self._orders[order.order_id] = order
        for rec in read_jsonl(self.fills_path):
            fill = Fill(**rec)
            self._fills.append(fill)
            if fill.kind == "close" and fill.order_id in self._orders:
                self._orders[fill.order_id].status = "closed"
                self.equity += fill.realized_pnl

    # -- BrokerBase ----------------------------------------------------------

    def create_order(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        target: float,
        size: float,
        risk_percent: float,
    ) -> PaperOrder:
        if direction not in ("long", "short"):
            raise ValueError(f"invalid direction: {direction}")
        now = _utcnow()
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            timestamp=now,
            symbol=symbol,
            direction=direction,
            size=size,
            entry=entry,
            stop=stop,
            target=target,
            risk_percent=risk_percent,
            status="open",
        )
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            timestamp=now,
            symbol=symbol,
            direction=direction,
            kind="open",
            price=entry,
            size=size,
        )
        append_jsonl(self.orders_path, order.model_dump(mode="json"))
        append_jsonl(self.fills_path, fill.model_dump(mode="json"))
        self._orders[order.order_id] = order
        self._fills.append(fill)
        return order

    def close_position(self, order_id: str, price: float) -> float:
        order = self._orders.get(order_id)
        if order is None or order.status != "open":
            raise ValueError(f"no open position for order_id {order_id}")
        if order.direction == "long":
            pnl = (price - order.entry) * order.size
        else:
            pnl = (order.entry - price) * order.size
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            timestamp=_utcnow(),
            symbol=order.symbol,
            direction=order.direction,
            kind="close",
            price=price,
            size=order.size,
            realized_pnl=pnl,
        )
        append_jsonl(self.fills_path, fill.model_dump(mode="json"))
        self._fills.append(fill)
        order.status = "closed"
        self.equity += pnl
        return pnl

    def open_positions(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.status == "open"]

    # -- Risk-state helpers ----------------------------------------------------

    def trades_today(self, symbol: str | None = None) -> int:
        today = _utcnow().date()
        return sum(
            1
            for o in self._orders.values()
            if o.timestamp.date() == today and (symbol is None or o.symbol == symbol)
        )

    def realized_pnl_pct(self, days: int) -> float:
        """Realized PnL over the last N days as a fraction of starting capital."""
        cutoff = _utcnow() - timedelta(days=days)
        pnl = sum(
            f.realized_pnl for f in self._fills if f.kind == "close" and f.timestamp >= cutoff
        )
        return pnl / self.starting_capital

    def daily_pnl_pct(self) -> float:
        return self.realized_pnl_pct(days=1)

    def weekly_pnl_pct(self) -> float:
        return self.realized_pnl_pct(days=7)
