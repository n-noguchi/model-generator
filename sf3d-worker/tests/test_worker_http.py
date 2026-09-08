import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parents[1]))
import worker


class _Response:
    def __init__(self, error=None):
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error


def test_post_returns_successful_response(monkeypatch):
    response = _Response()
    monkeypatch.setattr(worker.requests, "post", lambda *args, **kwargs: response)
    assert worker.post("/worker/jobs/claim") is response


def test_post_raises_for_api_error(monkeypatch):
    response = _Response(requests.HTTPError("500 Server Error"))
    monkeypatch.setattr(worker.requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(requests.HTTPError, match="500"):
        worker.post("/worker/jobs/complete")
