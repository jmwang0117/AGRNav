FROM nvidia/cudagl:11.4.2-devel-ubuntu18.04
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND noninteractive

# Basic dependencies and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    less \
    emacs \
    tmux \
    bash-completion \
    command-not-found \
    software-properties-common \
    xsel \
    xdg-user-dirs \
    python-pip \
    python-protobuf \
    python-apt \
    python-pexpect \
    pcl-tools \
    libomp5 \
    curl\
    locate\
    vulkan-utils


# ROS Melodic
RUN apt-get update
RUN apt-get install --assume-yes apt-utils
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-get update
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-desktop \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential

# Additional ROS package
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-derived-object-msgs \
    ros-melodic-ackermann-msgs \
    libarmadillo-dev \
    ros-melodic-pcl-conversions\
    ros-melodic-pcl-ros\
    ros-melodic-nlopt



RUN rosdep init
RUN rosdep update
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# Install tmux, vim and git
RUN apt-get install -y tmux vim git libsdl-image1.2-dev libsdl-dev libpcl-dev pcl-tools
RUN pip install pyyaml rospkg imageio
# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["bash"]
