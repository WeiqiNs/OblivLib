import argparse
import time

from oblivlib.dependency import AesGcm, AvlOmapConfig, Client, DaOramConfig, OramOstOmapConfig
from oblivlib.omap import AVLOmap, OramOstOmap
from oblivlib.oram import DAOram

KEY_SIZE = 16
DATA_SIZE = 16


def run_oram(client: Client, num_data: int) -> None:
    oram = DAOram(DaOramConfig(num_data=num_data, data_size=DATA_SIZE, client=client, encryptor=AesGcm()))
    oram.init_server_storage()

    start = time.perf_counter()
    for key in range(num_data):
        oram.operate_on_key(key, str(key).encode())
    errors = sum(oram.operate_on_key(key) != str(key).encode() for key in range(num_data))
    print(f"ORAM: {2 * num_data} accesses in {time.perf_counter() - start:.2f}s, {errors} errors")


def run_omap(client: Client, num_data: int) -> None:
    encryptor = AesGcm()
    ost = AVLOmap(
        AvlOmapConfig(num_data=num_data, key_size=KEY_SIZE, data_size=DATA_SIZE, client=client, encryptor=encryptor)
    )
    oram = DAOram(
        DaOramConfig(num_data=num_data, data_size=OramOstOmap.oram_data_size(ost), client=client, encryptor=encryptor)
    )
    omap = OramOstOmap(OramOstOmapConfig(num_data=num_data), ost=ost, oram=oram)
    omap.init_server_storage()

    start = time.perf_counter()
    for i in range(num_data):
        omap.insert(f"key-{i}".encode(), str(i).encode())
    errors = sum(omap.search(f"key-{i}".encode()) != str(i).encode() for i in range(num_data))
    print(f"OMAP: {2 * num_data} operations in {time.perf_counter() - start:.2f}s, {errors} errors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ORAM and an OMAP against a remote oblivlib server.")
    parser.add_argument("--endpoint", default="tcp://localhost:5555", help="server endpoint")
    parser.add_argument("--num-data", type=int, default=256, help="entries per structure")
    args = parser.parse_args()

    with Client.connect(args.endpoint) as client:
        run_oram(client, args.num_data)
        run_omap(client, args.num_data)
        print(f"Rounds: {client.metrics.rounds}, wire bytes: {client.metrics.wire_bytes}")


if __name__ == "__main__":
    main()
