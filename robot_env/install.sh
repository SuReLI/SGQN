pip install -U 'mujoco-py<2.2,>=2.1'
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mv mujoco210-linux-x86_64.tar.gz ~/.mujoco/
cd ~/.mujoco/
tar -xvf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/[USER_NAME]/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python -c "import mujoco_py"

