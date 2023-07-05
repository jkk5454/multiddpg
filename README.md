## Instructions for Installation
1. This codebase is tested with Ubuntu 20.04.6 LTS, CUDA Version: 12.0, Cuda compilation tools, release 11.8, V11.8.89 and Nvidia driver version 525.125.06. 

The following command will install some necessary dependencies.
```
sudo apt-get install build-essential libgl1-mesa-dev freeglut3-dev libglfw3 libgles2-mesa-dev
```

2. Create conda environment
   Create a conda environment and activate it: `conda env create -f environment.yml`

3. Using Docker to compile PyFlex.

  - Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
  - Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)
  - Install [Anaconda](https://www.anaconda.com/distribution/)
  - Install Pybind11 using `conda install pybind11`

  We posted the [Dockerfile](Dockerfile). To generate the pre-built file, download the Dockerfile in this directory and run
  ```
  docker build -t multiddpg .
  ```
in the directory that contains the Dockerfile.

  Using the following command to run docker
  ```
    nvidia-docker run \
    -v PATH_TO_SoftGym:/workspace/softgym \
    -v PATH_TO_CONDA:PATH_TO_CONDA \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it multiddpg bash
  ```
  Now you are in the Docker environment. Go to the softgym directory and compile PyFlex

  ```
  export PATH="PATH_TO_CONDA/bin:$PATH"
  . ./prepare_1.0.sh && ./compile_1.0.sh
  ```

4. Go to the root folder of MultiDDPG and run `. ./prepare_1.0.sh && ./compile_1.0.sh`. Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.

  ## MultiDDPG Train and Test
  Train MultiDDPG for 50 episodes as an example
  ```

  python examples/script.py --train 1
  ```

  Test MultiDDPG for 10 tasks with different inital positions
  ```

  python examples/script.py --train 0
  ```

  If the module 'softgym' can not be found. Please change the sys path in '.\examples\ddpg_env.py' and '.\exaples\ddpg_test.py'

## References
- NVIDIA FleX - 1.2.0: https://github.com/NVIDIAGameWorks/FleX
- Our python interface builds on top of PyFleX: https://github.com/YunzhuLi/PyFleX
- Softgym: https://github.com/Xingyu-Lin/softgym
