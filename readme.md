# Cell Estimator: A Web-Platform for Blood Cell Segmentation, Classification and Active Learning
Welcome to the project page of Group06 of the Applied Machine Learning course at the Technical University of Munich (TUM), Summer Semester 2023. Our project is called Cell Estimator. Our website offers blood cell segmentation, classification and relabeling functionality.

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

1. Clone this repository with this submodules:

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

After running this, the image should be pulled and the app should start on localhost:3000. Since the API takes more time to start than the frontend, it might be necessary to refresh the page in case you get fetch errors in the frontend. If you want to rerun after pulling, just run:

```
cd app
docker-compose up
```

Alternatively, you can build the image locally if you prefer. This takes however quite some time:

```
cd app
./build_and_run.sh
```

After this, you should be able to see the app running in localhost:3000.

## Using the deployed web application 

1. Login to the eduVPN
2. Visit https://group06.ami.dedyn.io and login with the credentials from the ami course.

This is the most reliable way of running the application, in case you are having problems running it locally with docker.

# Citation
If you use part of our code, please consider using the following citation:

@software{cellestimator2023,
  author = {Raimundo Becerra Parra and Konstantinos Larintzakis and Wafa Laroussi and Michael Lemanov and Ivan Nikolovski and Leonardo Fernandes Oliveira},
  title = {Cell Estimator: A Web-Platform for Blood Cell Segmentation, Classification and Active Learning},
  url = {https://github.com/RaiBP/cell-estimator},
  date = {2023-07-25},
}



