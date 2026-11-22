"""Cryptographic primitives: encryption and PRF implementations."""

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Final, override

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Encryptor(ABC):
    """Authenticated encryption of whole buckets.

    ``ciphertext_length(n)`` must be the exact length ``enc`` returns for an ``n``-byte plaintext: file
    storage sizes its fixed rows from it, and a longer ciphertext is rejected.
    """

    @abstractmethod
    def ciphertext_length(self, plaintext_length: int) -> int: ...

    @abstractmethod
    def enc(self, plaintext: bytes) -> bytes: ...

    @abstractmethod
    def dec(self, ciphertext: bytes) -> bytes: ...


class AesGcm(Encryptor):
    """AES-GCM with a fresh random 96-bit nonce per call: ``enc`` returns ``nonce ‖ ciphertext ‖ tag``,
    exactly 28 bytes longer than the plaintext. The key is random unless given. Random nonces cap safe
    use at about 2**32 encryptions per key. Instances are not picklable, and storage never keeps one."""

    NONCE_SIZE: Final = 12
    TAG_SIZE: Final = 16

    def __init__(self, key: bytes | None = None, key_byte_length: int = 16):
        if key_byte_length not in [16, 24, 32]:
            raise ValueError("The AES key length must be 16, 24, or 32 bytes.")

        self._aes_gcm: AESGCM = AESGCM(os.urandom(key_byte_length) if key is None else key)

    @override
    def ciphertext_length(self, plaintext_length: int) -> int:
        return self.NONCE_SIZE + plaintext_length + self.TAG_SIZE

    @override
    def enc(self, plaintext: bytes) -> bytes:
        """Encrypt under a fresh random nonce; returns nonce + ciphertext + tag."""
        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext_with_tag = self._aes_gcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext_with_tag

    @override
    def dec(self, ciphertext: bytes) -> bytes:
        """Decrypt a nonce + ciphertext + tag blob; raises InvalidTag if authentication fails."""
        nonce = ciphertext[: self.NONCE_SIZE]
        ciphertext_with_tag = ciphertext[self.NONCE_SIZE :]
        return self._aes_gcm.decrypt(nonce, ciphertext_with_tag, None)


class PseudoRandomFunction(ABC):
    @abstractmethod
    def digest(self, message: bytes) -> bytes: ...

    @abstractmethod
    def digest_mod_n(self, message: bytes, mod: int) -> int: ...


class Blake2Prf(PseudoRandomFunction):
    """Keyed BLAKE2b with a 64-byte digest. The 32-byte key is random unless given; schemes that must
    re-derive the same leaves (DA, Freecursive position maps) share one key."""

    KEY_SIZE: Final = 32
    DIGEST_SIZE: Final = 64

    def __init__(self, key: bytes | None = None):
        if key is not None and len(key) != self.KEY_SIZE:
            raise ValueError(f"The PRF key length must be {self.KEY_SIZE} bytes.")

        self._key: bytes = os.urandom(self.KEY_SIZE) if key is None else key

    @override
    def digest(self, message: bytes) -> bytes:
        return hashlib.blake2b(message, key=self._key, digest_size=self.DIGEST_SIZE).digest()

    @override
    def digest_mod_n(self, message: bytes, mod: int) -> int:
        return int.from_bytes(self.digest(message), "big") % mod


def hash_data_to_leaf(prf: PseudoRandomFunction, map_size: int, data: bytes) -> int:
    return prf.digest_mod_n(message=data, mod=map_size)


def hash_data_to_map(
    prf: PseudoRandomFunction, map_size: int, data: list[tuple[bytes, bytes]]
) -> dict[int, list[tuple[bytes, bytes]]]:
    data_map: dict[int, list[tuple[bytes, bytes]]] = {i: [] for i in range(map_size)}
    for pair in data:
        data_map[hash_data_to_leaf(prf=prf, map_size=map_size, data=pair[0])].append(pair)
    return data_map
