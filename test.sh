python3 train_rl.py --stock_filepaths 'data/1_4371167945.csv' \
    --obs_columns 'open' 'close' \
    --frames 1000000 --flag 'Debug' \
    --save_interval 100 --val_episode 100 \
    --seed 42 --arch 'attention'