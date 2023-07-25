# Group 06 - Cell estimator

Welcome to the project page of Group06. Our project is called Cell estimator. Our website offers blood cell segmentation, classification and relabeling functionality.

## Running the web application locally

### Requirements

1. Since we use segmentation algorithms that make use of the GPU, you will need a NVIDIA GPU in order to run it. The driver should support CUDA 12.0. Using the same hardware environment as the provided Kubernetes cluster works.
2. Docker
3. In order for your GPU to play nicely with docker, you need to install the nvidia container toolkit and restart your docker daemon

```
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Running the docker image

1. Clone this repository with its submodules:

```
git clone https://gitlab.lrz.de/ldv/teaching/ami/ami2023/projects/Group06.git --recurse-submodules
```

2. Login to the container registry:

```
docker login gitlab.lrz.de:5005
```

3. Run the app

```
cd app
./pull_and_run.sh
```

Alternatively, you can build the image locally if you prefer. This takes however quite some time:

```
cd app
./build_and_run.sh
```

After this, you should be able to see the app running in localhost:3000.

## Using the deployed web application 

1. Login to the eduVPN
2. Visit https://group06.ami.dedyn.io and login with the credentials below:

Username: ami
Password: ***REMOVED***



