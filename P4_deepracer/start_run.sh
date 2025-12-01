cd /home/ec2-user/cs-rl/P4_deepracer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate deepracer && nohup python - <<'PY' > train.log 2>&1 &
from src.run import run
run({})
PY
echo $! > train.pid