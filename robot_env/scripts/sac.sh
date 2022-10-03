export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/david.bertoin/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python src/train.py --algorithm sac --seed $1 --task_name reach --train_steps 250k;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/david.bertoin/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python src/train.py --algorithm sac --seed $1 --task_name hammerall --train_steps 250k;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/david.bertoin/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python src/train.py --algorithm sac --seed $1 --task_name push --train_steps 250k;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/david.bertoin/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python src/train.py --algorithm sac --seed $1 --task_name pegbox --train_steps 250k;


