
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsDefeatRandomEnemyShapedReward-v2 \
    --total-timesteps 100000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsDefeatWorkerRushEnemyShaped-v2 \
    --total-timesteps 100000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsDefeatLightRushEnemyShaped-v2 \
    --total-timesteps 100000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_autoregressive.py \
    --gym-id MicrortsDefeatWorkerRushEnemyHRL-v2 \
    --total-timesteps 100000000 \
    --wandb-project-name gym-microrts \
    --prod-mode \
    --wandb-entity vwxyzjn --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
