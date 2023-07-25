# Group 06 - Cell estimator

Welcome to the project page of Group06. Our project is called Cell estimator. Our website offers blood cell segmentation, classification and relabeling functionality.

## Running the web application

### Requirements

1. Since we use segmentation algorithms that make use of the GPU, you will need a NVIDIA GPU in order to run it. The driver should support CUDA 12.0. Using the same hardware environment as the provided Kubernetes cluster works.
2. Docker
3. In order for your GPU to play nicely with docker, you need to install the nvidia container toolkit and restart your docker daemon

```
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Running the docker image




