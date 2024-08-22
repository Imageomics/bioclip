import argparse
import sys
import tarfile
import webdataset as wds
from torch.utils.data import DataLoader

def log_and_continue(err):
    if isinstance(err, tarfile.ReadError) and len(err.args) == 3:
        print(err.args[2])
        return True
    if isinstance(err, ValueError):
        return True
    raise err

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shardlist", required=True, help="Path to all shards in braceexpand format."
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of worker processes"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--log-every", type=int, default=100, help="How often to log")
    args = parser.parse_args()

    keys = ("__key__", "jpg", "sci.txt", "com.txt", "sci_com.txt", "taxontag_com.txt")
    
    dataset = wds.DataPipeline(
        wds.SimpleShardList(args.shardlist),
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.decode("torchrgb"),
        wds.to_tuple(*keys, handler=log_and_continue),
    )
    
    dataloader = DataLoader(
        dataset, num_workers=args.workers, batch_size=args.batch_size
    )

    itr = iter(dataloader)
    batches = 0
    total_examples = 0
    while True:
        try:
            batch = next(itr)
            batch_size = len(batch[0])
            batches += 1
            total_examples += batch_size

            if batches % args.log_every == 0:
                eprint(f"{batches} batches / {total_examples} examples (current batch size: {batch_size})")
        except StopIteration:
            break

    eprint(f"Success! Processed {batches} batches with a total of {total_examples} examples.")
