# Some pointers

### Pytorch to deployment

- [Downscaling models](#Downscaling-models)
- Deployment through FLASK
    - resourceful tutorial <https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html>
    - which also points to how inputs can be batched for inference:
      [ShannonAI/service-streamer](https://github.com/ShannonAI/service-streamer),
      [Vision-Recognition-Service-with-Flask-and-service-streamer](https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer)
- ONXX: harware agnostic model
    - <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>
    - huggingface: <https://huggingface.co/transformers/serialization.html>,
      <https://github.com/huggingface/notebooks/blob/master/examples/onnx-export.ipynb>
- Torchscript: run in cpp env
    - <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>

### Downscaling models

- Quantization: improve inference time and memory footprint at (generally) small cost of accuracies
    - <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>
- TODO

### Upscaling data or dealing w/ less data

- TODO

### awesome-production-machine-learning

- <https://github.com/EthicalML/awesome-production-machine-learning>
- <https://github.com/eugeneyan/applied-ml>

### Using sqlite3 to load/query large files

- this repo <https://github.com/vzhong/embeddings> shows one use-case for Glove embeddings
- Mindmeld also loads queries using sqlite3
- related articles: <https://avi.im/blag/2021/fast-sqlite-inserts/>
    
