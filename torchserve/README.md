# Serve JointBERT with TorchServe

Create a .mar file
```bash
mkdir model_store
torch-model-archiver \
    --model-name JointBERT \
    --version 1.0 --serialized-file ../atis_model/pytorch_model.bin \
    --extra-files "../atis_model/config.json,../atis_model/training_args.bin,../data/atis/intent_label.txt,../data/atis/slot_label.txt,../atis_model/special_tokens_map.json,../atis_model/tokenizer_config.json,../atis_model/vocab.txt" \
    --handler "joint_intent_slot_serve_handler.py"
mv JointBERT.mar model_store
```

To serve with torchserve
```bash
torchserve --start --ncs --model-store model_store --models JointBERT.mar
```

Now you can send a POST request to the service
```bash
curl -X POST http://127.0.0.1:8080/predictions/JointBERT -T ../sample_pred_in.txt
```

Additionally, I have made a simple frontend under directory [flast](/flask)
