torch-model-archiver -f \
    --model-name JointBERT \
    --version 1.0 --serialized-file ./atis_model/pytorch_model.bin \
    --extra-files "./atis_model/config.json,./atis_model/training_args.bin,./data/atis/intent_label.txt,./data/atis/slot_label.txt,./atis_model/special_tokens_map.json,./atis_model/tokenizer_config.json,./atis_model/vocab.txt" \
    --handler "joint_intent_slot_serve_handler.py" --export-path "model_store"
torchserve --start --ncs --model-store model_store --models JointBERT.mar
    
