"""Intentionally non-functional live broker stub.

Live execution is out of scope for Milestone 1 and requires explicit human
approval (autonomy.yaml: submit_live_orders: false). This stub exists only to
mark where a demo-account adapter would plug in later. It stores no API keys.
"""

from execution.broker_base import BrokerBase


class OandaAdapterStub(BrokerBase):
    def create_order(self, *args, **kwargs):
        raise NotImplementedError(
            "Live/demo broker execution is intentionally not implemented in Milestone 1."
        )

    def close_position(self, *args, **kwargs):
        raise NotImplementedError(
            "Live/demo broker execution is intentionally not implemented in Milestone 1."
        )

    def open_positions(self):
        raise NotImplementedError(
            "Live/demo broker execution is intentionally not implemented in Milestone 1."
        )
