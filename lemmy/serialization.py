import abc
from pathlib import Path
from typing import Generic, TypeVar, Dict, Any, Union

import srsly
from spacy.util import ensure_path

C = TypeVar("C")


class Serializable(abc.ABC, Generic[C]):
    @classmethod
    def _version(cls) -> int:
        raise NotImplementedError

    def to_bytes(self) -> bytes:
        msg = self._to_bytes()
        msg["_version"] = self._version()
        return srsly.msgpack_dumps(msg)

    def _to_bytes(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_bytes(cls, bytes_data: bytes) -> C:
        msg = srsly.msgpack_loads(bytes_data)
        version = msg["_version"]
        if version != cls._version():
            raise TypeError(f"Incompatible versions: expected: {cls._version()} but got {version}")
        del msg["_version"]
        return cls._from_bytes(msg)

    @classmethod
    def _from_bytes(cls, msg: Dict[str, Any]) -> C:
        raise NotImplementedError

    def to_disk(self, path: Union[str, Path], exclude=tuple()) -> None:
        path = ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes())

    @classmethod
    def from_disk(cls, path: Union[str, Path], exclude=tuple()) -> C:
        path = ensure_path(path)
        with path.open("rb") as file_:
            obj = cls.from_bytes(file_.read())
            return obj
