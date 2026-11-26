# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host lifecycle for one injected standalone FFN backend."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

from afd_plugin.ffn_faas.interfaces import FFNBackend, FFNStepResult
from afd_plugin.ffn_faas.service import FFNService, ServicePhase

HOST_STATUS_POLL_SECONDS = 0.1


class FFNModelRunner:
    """Own the backend across stop requests and failed or pending receives."""

    def __init__(
        self,
        service: FFNService,
        backend_factory: Callable[[], FFNBackend],
        *,
        shutdown_timeout: float,
    ) -> None:
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown timeout must be finite and positive")
        self.service = service
        self.backend_factory = backend_factory
        self.shutdown_timeout = shutdown_timeout
        self.backend: FFNBackend | None = None
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._receiver_ready = threading.Event()
        self._receiver_continue = threading.Event()
        self._stop_lock = threading.Lock()
        self._backend_stop_requested = False
        self._failure: BaseException | None = None

    def _request_backend_stop(self) -> None:
        assert self.backend is not None
        with self._stop_lock:
            if self._backend_stop_requested:
                return
            self._backend_stop_requested = True
            self.backend.request_stop()

    def _serve(self) -> None:
        assert self.backend is not None
        try:
            self.backend.start_receiver()
            self._receiver_ready.set()
            self._receiver_continue.wait()
            while True:
                if self.service.stop_requested.is_set():
                    self._request_backend_stop()
                    break
                result = self.backend.execute_step()
                if not isinstance(result, FFNStepResult):
                    raise TypeError("FFN backend returned an invalid step result")
                if result is FFNStepResult.STOP:
                    break
                if result is FFNStepResult.WORK:
                    self.service.completed_batches()
                if self.service.stop_requested.is_set():
                    self._request_backend_stop()
                    break
            self.backend.receiver_exited()
        except BaseException as error:
            self._failure = error
            self._receiver_ready.set()
        finally:
            self._done.set()

    def _join_receiver(self) -> None:
        assert self._thread is not None
        self._thread.join(timeout=self.shutdown_timeout)
        if self._thread.is_alive():
            raise RuntimeError(
                "FFN receive is pending; retain owners and isolate the "
                "deployment. A host stop cannot cancel backend device work."
            )
        if self._failure is not None:
            raise RuntimeError("FFN receiver failed; retain resources") from self._failure

    def run(self) -> None:
        """Activate the receiver before READY and retain owners on failure."""
        try:
            if self.service.stop_requested.is_set():
                self.service.phase(ServicePhase.STOPPED, resources_released=True)
                return
            self.service.phase(ServicePhase.LOADING)
            self.backend = self.backend_factory()
            self.backend.prepare(self.service)
            self._thread = threading.Thread(
                target=self._serve, name="ffn-receiver", daemon=True
            )
            self._thread.start()
            if not self._receiver_ready.wait(timeout=self.shutdown_timeout):
                self.service.request_stop()
                self._receiver_continue.set()
                raise RuntimeError("FFN receiver did not become executable in time")
            if self._failure is not None:
                raise RuntimeError("FFN receiver activation failed") from self._failure

            if self.service.stop_requested.is_set():
                self._request_backend_stop()
            else:
                try:
                    self.service.phase(ServicePhase.READY)
                except InterruptedError:
                    self._request_backend_stop()
            self._receiver_continue.set()
            while not self._done.is_set():
                if self.service.stop_requested.wait(HOST_STATUS_POLL_SECONDS):
                    if not self._done.is_set():
                        self._request_backend_stop()
                    break
            self._join_receiver()
            self.backend.close()
            self.service.phase(ServicePhase.STOPPED, resources_released=True)
        except BaseException as error:
            # No native close on a failed startup/receive: an enqueued task may
            # still reference weights, Graph pools or communication buffers.
            self.service.phase(ServicePhase.FAILED, error=str(error))
            raise
