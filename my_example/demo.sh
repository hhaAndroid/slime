set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export PYTHONUNBUFFERED=1

if [ "$RANK" -eq 0 ]; then
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT
    sleep 30
else
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --block
fi
sleep 30

RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir . -- demo_1.py 2>&1 | tee verl_demo.log
