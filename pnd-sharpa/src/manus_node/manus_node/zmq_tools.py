from __future__ import annotations

from ctypes import CDLL, byref, c_char_p, c_int, c_size_t, c_void_p
from ctypes import cast, create_string_buffer, sizeof
from ctypes.util import find_library


ZMQ_SUB = 2
ZMQ_SUBSCRIBE = 6
ZMQ_LINGER = 17
ZMQ_RCVHWM = 24
ZMQ_DONTWAIT = 1
EAGAIN = 11


class NonBlockingSubscriber:
    """ZMQ SUB socket with pyzmq first and libzmq ctypes fallback."""

    def __init__(self, address: str, receive_hwm: int = 1) -> None:
        self.address = address
        self.backend = "pyzmq"
        self._py_socket = None
        self._py_context = None
        self._py_again = None
        self._czmq: CtypesZmqSubscriber | None = None
        try:
            import zmq  # type: ignore

            self._py_context = zmq.Context.instance()
            self._py_socket = self._py_context.socket(zmq.SUB)
            self._py_socket.setsockopt(zmq.RCVHWM, receive_hwm)
            self._py_socket.setsockopt(zmq.LINGER, 0)
            self._py_socket.connect(address)
            self._py_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._py_again = zmq.Again
            return
        except ModuleNotFoundError:
            self.backend = "ctypes-libzmq"
        except Exception:
            self.close()
            raise

        self._czmq = CtypesZmqSubscriber(address, receive_hwm)

    def recv_nonblocking(self) -> bytes | None:
        if self._py_socket is not None:
            try:
                return self._py_socket.recv(flags=ZMQ_DONTWAIT)
            except self._py_again:  # type: ignore[misc]
                return None

        if self._czmq is None:
            return None
        return self._czmq.recv_nonblocking()

    def close(self) -> None:
        if self._py_socket is not None:
            self._py_socket.close(linger=0)
            self._py_socket = None
        if self._czmq is not None:
            self._czmq.close()
            self._czmq = None


class CtypesZmqSubscriber:
    def __init__(self, address: str, receive_hwm: int = 1) -> None:
        lib_path = find_library("zmq") or "libzmq.so.5"
        self.lib = CDLL(lib_path)
        self._configure_api()
        self.context = self.lib.zmq_ctx_new()
        if not self.context:
            raise RuntimeError("zmq_ctx_new failed")
        self.socket = self.lib.zmq_socket(self.context, ZMQ_SUB)
        if not self.socket:
            self.lib.zmq_ctx_term(self.context)
            raise RuntimeError("zmq_socket SUB failed")

        try:
            self._set_int(ZMQ_RCVHWM, receive_hwm)
            self._set_int(ZMQ_LINGER, 0)
            self._set_bytes(ZMQ_SUBSCRIBE, b"")
            if self.lib.zmq_connect(self.socket, address.encode("utf-8")) != 0:
                raise RuntimeError(f"zmq_connect failed: {self._last_error()}")
        except Exception:
            self.close()
            raise

    def _configure_api(self) -> None:
        self.lib.zmq_ctx_new.restype = c_void_p
        self.lib.zmq_ctx_term.argtypes = [c_void_p]
        self.lib.zmq_socket.argtypes = [c_void_p, c_int]
        self.lib.zmq_socket.restype = c_void_p
        self.lib.zmq_setsockopt.argtypes = [c_void_p, c_int, c_void_p, c_size_t]
        self.lib.zmq_connect.argtypes = [c_void_p, c_char_p]
        self.lib.zmq_recv.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        self.lib.zmq_recv.restype = c_int
        self.lib.zmq_close.argtypes = [c_void_p]
        self.lib.zmq_errno.restype = c_int
        self.lib.zmq_strerror.argtypes = [c_int]
        self.lib.zmq_strerror.restype = c_char_p

    def _set_int(self, option: int, value: int) -> None:
        c_value = c_int(value)
        result = self.lib.zmq_setsockopt(
            self.socket,
            option,
            cast(byref(c_value), c_void_p),
            c_size_t(sizeof(c_value)),
        )
        if result != 0:
            raise RuntimeError(f"zmq_setsockopt({option}) failed: {self._last_error()}")

    def _set_bytes(self, option: int, value: bytes) -> None:
        buf = create_string_buffer(value)
        result = self.lib.zmq_setsockopt(
            self.socket,
            option,
            cast(buf, c_void_p),
            c_size_t(len(value)),
        )
        if result != 0:
            raise RuntimeError(f"zmq_setsockopt({option}) failed: {self._last_error()}")

    def recv_nonblocking(self) -> bytes | None:
        buf = create_string_buffer(1024 * 1024)
        size = self.lib.zmq_recv(self.socket, buf, c_size_t(len(buf)), ZMQ_DONTWAIT)
        if size >= 0:
            return bytes(buf.raw[:size])
        if self.lib.zmq_errno() == EAGAIN:
            return None
        raise RuntimeError(f"zmq_recv failed: {self._last_error()}")

    def _last_error(self) -> str:
        errno = self.lib.zmq_errno()
        message = self.lib.zmq_strerror(errno)
        return message.decode("utf-8", errors="replace") if message else str(errno)

    def close(self) -> None:
        socket = getattr(self, "socket", None)
        if socket:
            self.lib.zmq_close(socket)
            self.socket = None
        context = getattr(self, "context", None)
        if context:
            self.lib.zmq_ctx_term(context)
            self.context = None
