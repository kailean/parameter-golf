"""Upload training data to Modal volume."""
import modal

app = modal.App("parameter-golf-data-upload")
vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.local_entrypoint()
def main():
    """Upload training data files to Modal volume."""
    print("Uploading data to Modal volume...")
    
    with vol.batch_upload() as batch:
        # SP8192 tokenizer
        batch.put_file(
            "data/tokenizers/fineweb_8192_bpe.model",
            "data/tokenizers/fineweb_8192_bpe.model"
        )
        # SP1024 tokenizer
        batch.put_file(
            "/Volumes/MacStorageExtended/parameter-golf-data/sp8192_datasets/fineweb_1024_bpe.model",
            "data/tokenizers/fineweb_1024_bpe.model"
        )
        # SP8192 val data
        batch.put_file(
            "/Volumes/MacStorageExtended/parameter-golf-data/sp8192_datasets/fineweb_val_000000.bin",
            "data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
        )
        # SP8192 train shard 0
        batch.put_file(
            "/Volumes/MacStorageExtended/parameter-golf-data/sp8192_datasets/fineweb_train_000000.bin",
            "data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin"
        )
    
    print("✅ Data uploaded to Modal volume!")

