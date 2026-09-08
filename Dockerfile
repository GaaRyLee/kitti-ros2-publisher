FROM ros:humble-ros-base-jammy

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-humble-cv-bridge \
    ros-humble-geometry-msgs \
    ros-humble-rviz2 \
    ros-humble-sensor-msgs-py \
    ros-humble-tf2-ros \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CUDA-enabled PyTorch.  The NVIDIA driver is supplied by the host through
# Docker's GPU runtime; CUDA toolkit installation in this image is unnecessary.
RUN python3 -m pip install --no-cache-dir \
        'numpy<2' \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && python3 -m pip install --no-cache-dir \
        torch-cluster \
        -f https://data.pyg.org/whl/torch-2.3.1+cu118.html

# torch-cluster's resolver may otherwise replace the ROS/OpenCV-compatible NumPy.
RUN python3 -m pip install --no-cache-dir --force-reinstall 'numpy<2'

# Make every interactive shell ROS-ready.  The workspace setup is available
# only after colcon build has created /ros2_ws/install/setup.bash.
RUN printf '%s\n' \
    'source /opt/ros/humble/setup.bash' \
    'if [ -f /ros2_ws/install/setup.bash ]; then' \
    '  source /ros2_ws/install/setup.bash' \
    'fi' \
    >> /root/.bashrc
