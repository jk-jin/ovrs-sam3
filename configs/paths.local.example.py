# Copy to configs/paths.local.py and set machine-local paths.
# Do not commit real credentials or machine-specific absolute paths.
_base_ = ["experiments/main/isaid_loveda_full.py"]

runtime = dict(
    work_dir="./work_dirs/local/isaid_loveda_full",
)

model = dict(
    bpe_path="/path/to/assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="/path/to/weights/sam3.pt",
    remoteclip=dict(
        pretrained="/path/to/weights/RemoteCLIP-ViT-L-14.pt",
    ),
)

train_dataloader = dict(
    dataset=dict(
        img_dir="/path/to/iSAID/img_dir/train",
        ann_dir="/path/to/iSAID/ann_dir/train",
    ),
)

val_dataloader = dict(
    dataset=dict(
        img_dir="/path/to/loveDA/img_dir/val",
        ann_dir="/path/to/loveDA/ann_dir/val",
    ),
)
