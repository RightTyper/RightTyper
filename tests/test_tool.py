import sys
from righttyper.righttyper_tool import setup_tool_id, reset_monitoring


def test_reset_monitoring_before_init():
    reset_monitoring()  # should not throw


def test_reset_monitoring_after_init():
    setup_tool_id()
    reset_monitoring()
