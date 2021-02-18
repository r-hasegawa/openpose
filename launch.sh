sudo xhost + local:root
sudo docker run -it --gpus all -e DISPLAY=$DISPLAY --net host -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority  --device=/dev/video0:/dev/video0:mwr -v $HOME/openpose:/openpose -v $HOME/data:/data cnjoads/openpose_staf:20201211 /bin/bash
