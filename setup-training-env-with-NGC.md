---
title: setup training env with NGC
date: 2021-06-04 16:45:56
tags: ngc
---

## setup 

[ngc](https://ngc.nvidia.com/) requires only host GPU driver, SDK, such as cuda, tf is not necessary required on host machine. while for an easy demo test, it's helpful to locally have cuda, tf.


* [install cuda locally](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

* system summary (gpu driver 465, cuda version 11.3)

* install docker engine (20.10) 

* [add nvidia-docker-runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#adding-the-nvidia-runtime)

```sh
* setup /etc/systemd/system/docker.service.d/override.conf
* setup /etc/docker/daemon.json 
sudo systemctl restart docker
```

* [install nvidia-docker cli](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

```sh
* setup /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update 
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

* nvidia-docker test 

```sh
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host nvcr.io/nvidia/tensorflow:21.05-tf2-py3
```

* install tensorflow locally 

[pip install tf](https://www.tensorflow.org/install/pip)

```sh
#active python venv 
source ./venv/bin/activate
pip install --upgrade tensorflow
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# exit venv
deactivate
```

* install tfds locally

```sh
export TFDS_DIR=~/data/tensorflow_datasets
tfds build "cifar_10" 
```

* launch jupyter on bare system

```sh
pip install jupyterlab
export PATH="$PATH:$HOME/.local/bin"
jupyter kernelspec uninstall venv  #uninstall the new created kernel

```

* launch jupyter in virtualenv 

[zhihu](https://zhuanlan.zhihu.com/p/33257881)


```sh
source venv/bin/activate
python -m pip install ipykernel
python -m ipykernel install --user --name=venv
jupyter notebook  #add .local/bin to venv/bin/activate 
```


* [how to start notebook on a custom IP or port](https://jupyter.readthedocs.io/en/latest/running.html#running)

```sh 
jupyter notebook --port 9999
```



* install ngc 

```sh
ngc registry resource download-version "nvidia/efficientnet_for_tensorflow2:21.02.1"
```



## first look at NGC EfficientNet

check efficientNet from [nvidia github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets/efficientnet), or from [nvidia ngc](https://ngc.nvidia.com/catalog/resources/nvidia:efficientnet_for_tensorflow2/)


* manually download ImageNet 2012, taking hours

* to build a bare EfficientNet docker image by:


```sh
cd ~/efficientnet_for_tensorflow2_v21.02.1
bash ./scripts/docker/build.sh
```

which add dependent modules based on tf2 base image.


* launch this docker image by :

```sh
bash ./scripts/docker/launch.sh
```

what's inside the `launch.sh` simpely:

```sh
nvidia-docker run -it --rm --net=host -v host_vol:container_vol --runtime=nvidia nvcr.io/nvidia/efficientnet-tf2:21.05-tf2-py3
```

basically, we start the docker container as interactive mode, so as like user login to a virtual machine, and do training works inside. 


* start training inside the launched container

the following is sample to train B0 with TF32 on ImageNet(ILSVRC2012),

```sh
.~/efficientnet_for_tensorflow2_v21.02.1/scripts/B0/training/TF32/convergence_8XA100-80G.sh
```

the `convergence_xx.sh` trig `horovodrun` script. 

```sh
horovodrun -np \# bash ./scripts/bind.sh --cpu=exclusive --ib=single -- python3 main.py 
```

* bind.sh

inside which run [numactl](https://www.systutorials.com/docs/linux/man/8-numactl/) to control binding policy for each task. 


```sh
numactl --show #show numa policy setting of current process 
numactl --hardware #show available nodes 
numactl --physcpubind=cpus, --membind=nodes python my.py
```


* run demo horovodrun with bind.sh:

```sh
horovodrun -np 1 bash ./scripts/bind.sh --cpu=exclusive --mem=off -- python3 hello.py
```

please launch the docker with the suggested arguments, if not `membind` report errors as following:

```sh
[1,0]<stderr>:+ exec numactl --physcpubind=0-7,8-15 --membind=0 -- python3 test.py
[1,0]<stderr>:set_mempolicy: Operation not permitted
[1,0]<stderr>:setting membind: Operation not permitted
```


next to:

* train EfficientNet with imageNet2012


* image classification with pre-trained efficientNet on your own dataset 







## reference 

* [ngc user guide](https://docs.nvidia.com/dgx/ngc-registry-for-dgx-user-guide/)

* [DL frameworks doc](https://docs.nvidia.com/deeplearning/frameworks/index.html)


