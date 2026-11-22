import random
from typing import Any

import pytest

from oblivlib.dependency import ContractError, DaOramConfig, FreecursiveOramConfig, RecursiveOramConfig
from oblivlib.dependency.types import UNSET
from oblivlib.oram import DAOram, FreecursiveOram, RecursivePathOram


class TestOramCommon:
    @pytest.mark.parametrize("storage", ["plain", "encrypted", "file"])
    def test_init_with_data_serves_every_key(self, make_oram, client, tmp_path, encryptor, storage):
        build_file = tmp_path / "build.tree"
        config: dict[str, Any] = {
            "plain": {},
            "encrypted": {"encryptor": encryptor},
            "file": {"encryptor": encryptor, "build_file": build_file},
        }[storage]
        oram = make_oram(num_data=100, data_size=10, client=client, **config)
        oram.init_server_storage(data=((key, b"v%d" % key) for key in range(100) if key % 3))

        assert not build_file.exists()
        assert [oram.operate_on_key(key) for key in range(100)] == [
            b"v%d" % key if key % 3 else b"" for key in range(100)
        ]

    @pytest.mark.parametrize("num_data", [100, 1025])
    def test_reads_writes_and_updates_match_a_dict_model(self, make_oram, client, duplicate_blocks, num_data):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        model = {key: b"i%d" % key if key % 2 else b"" for key in range(num_data)}
        oram.init_server_storage(data={key: value for key, value in model.items() if value})
        rng = random.Random(num_data)

        key = 0
        for step in range(num_data * 4):
            key = key if rng.random() < 0.25 else rng.randrange(num_data)
            new = rng.choice([b"", b"%d" % step])
            match rng.choice(["read", "write", "update"]):
                case "read":
                    assert oram.operate_on_key(key) == model[key]
                case "write":
                    assert oram.operate_on_key(key, new) == model[key]
                    model[key] = new
                case _:
                    assert oram.update(key, lambda old, new=new: (new, old + b"!")) == model[key] + b"!"
                    model[key] = new
        assert [oram.operate_on_key(key) for key in range(num_data)] == list(model.values())
        assert not duplicate_blocks(oram)

    def test_contract_violations_raise_before_any_message(self, make_oram, recorded_client):
        client, recorder = recorded_client()
        oram = make_oram(num_data=64, data_size=10, client=client)
        init_cases: list[tuple[list[tuple[Any, Any]], str]] = [
            ([(1, b"a"), (1, b"b")], "given twice"),
            ([(64, b"a")], r"\[0, 64\)"),
            ([(-1, b"a")], r"\[0, 64\)"),
            ([(1, "a")], "must be bytes"),
            ([(1, b"x" * 11)], "data_size is 10"),
        ]
        for pairs, match in init_cases:
            with pytest.raises(ContractError, match=match):
                oram.init_server_storage(data=iter(pairs))
        assert not recorder.messages

        oram.init_server_storage(data={5: b"kept"})
        sent = len(recorder.messages)
        access_cases: list[tuple[Any, Any]] = [
            (5, "hello"),
            (5, b"x" * 11),
            (5, None),
            (-1, b"v"),
            (64, b"v"),
            ("0", b"v"),
            (True, b"v"),
        ]
        for key, value in access_cases:
            with pytest.raises(ContractError, match=f"{type(oram).__name__} '"):
                oram.operate_on_key(key=key, value=value)
        for key in (-1, 64, "0", True):
            with pytest.raises(ContractError, match=f"{type(oram).__name__} '"):
                oram.update(key, lambda old: pytest.fail("change ran"))

        assert len(recorder.messages) == sent
        assert oram.operate_on_key(key=5) == b"kept"

    @pytest.mark.parametrize("new", ["v", b"x" * 11], ids=["str", "oversized"])
    def test_update_rejects_a_value_its_change_returns_out_of_contract(self, make_oram, client, new):
        oram = make_oram(num_data=64, data_size=10, client=client)
        oram.init_server_storage()
        with pytest.raises(ContractError, match=f"{type(oram).__name__} '"):
            oram.update(3, lambda old: (new, None))

    def test_an_access_begun_inside_another_is_refused_before_any_message(self, make_oram, recorded_client):
        client, recorder = recorded_client()
        oram = make_oram(num_data=64, data_size=10, client=client)
        oram.init_server_storage()
        sent: list[int] = []

        def nested(old):
            sent.append(len(recorder.messages))
            oram.operate_on_key(2)
            return old, None

        with pytest.raises(ContractError, match="while another one was still running"):
            oram.update(1, nested)
        assert len(recorder.messages) == sent[0]

    def test_write_deferral_preserves_access_sequence(self, make_oram, assert_deferral_preserves_access):
        def workload(client):
            oram = make_oram(num_data=64, data_size=10, client=client)
            oram.init_server_storage()
            rng = random.Random(5)
            for step in range(60):
                oram.operate_on_key(key=rng.randrange(64), value=b"%d" % step if step % 2 else UNSET)
            oram.update(3, lambda old: (b"1", None))

        rounds_off, rounds_on = assert_deferral_preserves_access(workload)
        assert rounds_on * 2 == rounds_off


@pytest.mark.parametrize(
    ("make", "level_count"),
    [
        (lambda client: RecursivePathOram(RecursiveOramConfig(num_data=256, data_size=10, client=client)), 3),
        (lambda client: DAOram(DaOramConfig(num_data=256, data_size=10, client=client, num_ic=4)), 2),
        (lambda client: FreecursiveOram(FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4)), 2),
    ],
    ids=["recursive", "da", "freecursive"],
)
def test_max_stash_adds_every_level_peak(make, level_count, client):
    oram = make(client)
    oram.init_server_storage()
    for key in range(0, 256, 9):
        oram.operate_on_key(key, b"v")

    assert len(oram._levels) == level_count
    assert all(level._max_stash >= 1 for level in oram._levels)
    assert oram.max_stash > oram._max_stash


_RESETTING_COUNTER_ORAMS = pytest.mark.parametrize(
    "make_counter_oram",
    [
        lambda client: DAOram(DaOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=2)),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=4, reset_prob=0.5)
        ),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=2, reset_method="hard")
        ),
    ],
    ids=["da", "freecursive_prob", "freecursive_hard"],
)


@_RESETTING_COUNTER_ORAMS
def test_write_deferral_preserves_counter_resets(make_counter_oram, assert_deferral_preserves_access):
    def workload(client):
        oram = make_counter_oram(client)
        oram.init_server_storage()
        rng = random.Random(6)
        for step in range(120):
            oram.operate_on_key(key=rng.randrange(256), value=b"%d" % step)
        for key in range(4):
            oram.update(key, lambda old, key=key: (b"%d" % key, None))

    assert_deferral_preserves_access(workload)


@_RESETTING_COUNTER_ORAMS
def test_updates_through_counter_resets_match_a_dict_model(make_counter_oram, client, duplicate_blocks):
    oram = make_counter_oram(client)
    oram.init_server_storage()
    model = dict.fromkeys(range(16), b"")
    rng = random.Random(9)
    for step in range(400):
        key = rng.randrange(16)
        assert oram.update(key, lambda old, step=step: (b"%d" % step, old)) == model[key]
        model[key] = b"%d" % step
    assert [oram.operate_on_key(key) for key in range(16)] == list(model.values())
    assert not duplicate_blocks(oram)


@pytest.mark.parametrize("mode", ["adopt", "attach", "stream"])
def test_every_build_file_handover_mode_serves_the_initial_data(mode, tmp_path, encryptor, handover_client):
    client = handover_client(mode)
    build_file = tmp_path / "build.tree"
    oram = RecursivePathOram(
        RecursiveOramConfig(num_data=128, data_size=10, client=client, encryptor=encryptor, build_file=build_file)
    )
    with client.measure() as init:
        oram.init_server_storage(data={key: b"%d" % (key * 3) for key in range(128)})

    assert not build_file.exists()
    assert init.result.rows_sealed > 0
    assert (init.result.row_bytes_written > 0) is (mode == "stream")
    for key in range(128):
        assert oram.operate_on_key(key=key, value=b"%d" % key) == b"%d" % (key * 3)
    for key in range(128):
        assert oram.operate_on_key(key=key) == b"%d" % key
    client.close()
