import argparse

from oblivlib.dependency import AesGcm, AvlOmapConfig, Client, DaOramConfig, Measurement
from oblivlib.omap import AVLOmap, OramOstOmap
from oblivlib.oram import DAOram

KEY_SIZE = 16
DATA_SIZE = 16
COLUMNS = (
    "rounds",
    "sent KB",
    "received KB",
    "server ms",
    "network+reply ms",
    "wire ms",
    "codec ms",
    "crypto ms",
    "client ms",
)


def _row(name: str, measurement: Measurement) -> str:
    spent = measurement.result
    values = (
        spent.bytes_sent / 1e3,
        spent.bytes_received / 1e3,
        spent.server_ns / 1e6,
        spent.network_ns / 1e6,
        spent.client_wire_ns / 1e6,
        (spent.encode_ns + spent.decode_ns) / 1e6,
        spent.crypto_ns / 1e6,
        measurement.client_ns / 1e6,
    )
    return f"{name:<6}{spent.rounds:>13}" + "".join(f"{value:>13.1f}" for value in values)


def run_oram(client: Client, num_data: int) -> Measurement:
    oram = DAOram(DaOramConfig(num_data=num_data, data_size=DATA_SIZE, client=client, encryptor=AesGcm()))
    oram.init_server_storage()

    with client.measure() as measurement:
        for key in range(num_data):
            oram.operate_on_key(key, str(key).encode())
        errors = sum(oram.operate_on_key(key) != str(key).encode() for key in range(num_data))
        client.flush()
    print(f"ORAM: {2 * num_data} accesses in {measurement.elapsed_ns / 1e9:.2f}s, {errors} errors")
    return measurement


def run_omap(client: Client, num_data: int) -> Measurement:
    encryptor = AesGcm()
    ost = AvlOmapConfig(num_data=num_data, key_size=KEY_SIZE, data_size=DATA_SIZE, client=client, encryptor=encryptor)
    oram = DAOram(
        DaOramConfig(
            num_data=num_data, data_size=OramOstOmap.oram_data_size(AVLOmap, ost), client=client, encryptor=encryptor
        )
    )
    omap = OramOstOmap(ost=AVLOmap, ost_config=ost, oram=oram)
    omap.init_server_storage()

    with client.measure() as measurement:
        for i in range(num_data):
            omap.insert(f"key-{i}".encode(), str(i).encode())
        errors = sum(omap.search(f"key-{i}".encode()) != str(i).encode() for i in range(num_data))
        client.flush()
    print(f"OMAP: {2 * num_data} operations in {measurement.elapsed_ns / 1e9:.2f}s, {errors} errors")
    return measurement


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ORAM and an OMAP against a remote oblivlib server.")
    parser.add_argument("--endpoint", default="tcp://localhost:5555", help="server endpoint")
    parser.add_argument("--num-data", type=int, default=256, help="entries per structure")
    args = parser.parse_args()

    with Client.connect(args.endpoint) as client:
        oram = run_oram(client, args.num_data)
        omap = run_omap(client, args.num_data)
    print(f"{'':<6}" + "".join(f"{column:>13}" for column in COLUMNS))
    print(_row("ORAM", oram))
    print(_row("OMAP", omap))


if __name__ == "__main__":
    main()
