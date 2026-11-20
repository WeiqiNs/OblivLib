import os

import pytest
from cryptography.exceptions import InvalidTag

from oblivlib.dependency import AesGcm, Blake2Prf, FeistelPrp


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


class TestPrp:
    @pytest.mark.parametrize("domain_size", [2, 37])
    def test_bijection_with_inverse(self, domain_size):
        prp = FeistelPrp(domain_size=domain_size)
        outputs = [prp.permute(x) for x in range(domain_size)]
        assert sorted(outputs) == list(range(domain_size))
        assert [prp.inverse(y) for y in outputs] == list(range(domain_size))

    def test_deterministic_with_fixed_key(self):
        key = os.urandom(16)
        prp_a = FeistelPrp(domain_size=100, key=key)
        prp_b = FeistelPrp(domain_size=100, key=key)
        assert [prp_a.permute(x) for x in range(100)] == [prp_b.permute(x) for x in range(100)]

    def test_different_keys_differ(self):
        prp_a = FeistelPrp(domain_size=100)
        prp_b = FeistelPrp(domain_size=100)
        assert [prp_a.permute(x) for x in range(100)] != [prp_b.permute(x) for x in range(100)]

    def test_rejects_bad_parameters_and_inputs(self):
        with pytest.raises(ValueError, match="Domain size"):
            FeistelPrp(domain_size=1)
        with pytest.raises(ValueError, match=r"at most 2\*\*512"):
            FeistelPrp(domain_size=2**512 + 1)
        widest = FeistelPrp(domain_size=2**512, key=bytes(16))
        assert widest.inverse(widest.permute(5)) == 5
        with pytest.raises(ValueError, match="at least 16 bytes"):
            FeistelPrp(domain_size=10, key=bytes(8))
        prp = FeistelPrp(domain_size=10)
        for bad in (-1, 10, 11):
            with pytest.raises(ValueError):
                prp.permute(bad)
            with pytest.raises(ValueError):
                prp.inverse(bad)
