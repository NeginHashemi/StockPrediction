python3 train_rl.py --stock_filepaths 'data/1_4371167945.csv' \
    --obs_columns 'open' 'close' 'time' \
    --frames 1000000 --flag 'Debug' \
    --save_interval 10 --val_episode 10