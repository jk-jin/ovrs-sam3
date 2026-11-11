model = dict(
    task_mode="semantic",
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="weights/sam3.pt",
    load_from_hf=False,
    eval_mode=False,
    compile=False,
    prompt_chunk_size=4,

    remoteclip=dict(
        enabled=True,
        model_name="ViT-L-14",
        pretrained="weights/RemoteCLIP-ViT-L-14.pt",
        default_output="feat_map",
        image_size=504,
        image_intermediate_layers=[7, 15],
        normalize_label_for_clip=True,
        text_prompt_batch_size=64,
        text_prompt_use_checkpoint=True,
    ),

    aggregation=dict(
        enabled=True,
        num_blocks=4,  # cross -> intra -> cross -> intra; total blocks.
        num_heads=8,
        dropout=0.1,
        feature_dim=256,
        semantic_dim=256,
        aggregation_hw=36,
        encoder_hw=72,
        window_size=12,
        shift_size=6,
        use_checkpoint=True,
    ),

    freeze=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.feature_aggregation",
        ],
        frozen_modules=[],
        remoteclip_text_finetune="attention",
        remoteclip_image_finetune="attention",
    ),
)
