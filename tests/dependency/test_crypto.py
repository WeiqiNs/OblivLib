import os

import pytest
from cryptography.exceptions import InvalidTag

from oblivlib.dependency import AesGcm, Blake2Prf


class TestAesGcm:
    def test_round_trip_for_every_key_size(self):
        for key_size in [16, 24, 32]:
            aesgcm = AesGcm(key_byte_length=key_size)
            assert aesgcm.dec(aesgcm.enc(b"Hello")) == b"Hello"

    def test_ciphertext_length(self):
        aesgcm = AesGcm()
        for length in [0, 1, 15, 16, 17, 32, 100]:
            assert aesgcm.ciphertext_length(length) == len(aesgcm.enc(os.urandom(length)))

    def test_invalid_key_size_raises(self):
        with pytest.raises(ValueError):
            AesGcm(key_byte_length=20)

    def test_tampered_ciphertext_raises(self):
        aesgcm = AesGcm()
        ciphertext = bytearray(aesgcm.enc(b"authentic message"))
        ciphertext[-1] ^= 0x01
        with pytest.raises(InvalidTag):
            aesgcm.dec(bytes(ciphertext))

    def test_distinct_nonce_per_encryption(self):
        aesgcm = AesGcm()
        assert aesgcm.enc(b"same") != aesgcm.enc(b"same")


class TestPrf:
    def test_same_key_same_digest(self):
        key = os.urandom(Blake2Prf.KEY_SIZE)
        assert Blake2Prf(key=key).digest(b"msg") == Blake2Prf(key=key).digest(b"msg")

    def test_digest_mod_n_in_range(self):
        prf = Blake2Prf()
        for i in range(100):
            assert 0 <= prf.digest_mod_n(message=str(i).encode(), mod=17) < 17

    def test_wrong_key_length_raises(self):
        with pytest.raises(ValueError, match="32 bytes"):
            Blake2Prf(key=bytes(16))
