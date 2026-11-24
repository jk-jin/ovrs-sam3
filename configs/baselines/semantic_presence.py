_base_ = ["../test/loveda.py"]
model = dict(pipeline="sam3_semantic_presence", remoteclip=dict(enabled=False), aggregation=dict(enabled=False),
    freeze=dict(train_adapters_only=False, trainable_modules=[], frozen_modules=["core"],
        remoteclip_text_finetune="frozen", remoteclip_image_finetune="frozen"))
experiment = dict(name="semantic_presence", method_id="sam3_semantic", method_name="SAM 3",
    variant_id="semantic_presence", implementation_id="sam3-original-semantic-v1", kind="baseline")
training = dict(regime="training_free", dataset="none", split=None, protocol_id="sam3_original_pretrained")

loss = dict(reference_distill_weight=0.0)
