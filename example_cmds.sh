
# For a D4RL environment
python run_full_training.py --env d4rl_env_maker.antmaze_umaze -off bc -on sac

# For an existing offline dataset and trained guide
python run_full_training.py --env CartPole-v0 -on dqn --offline_data_path "offline_data/CartPole-v1_PGConfig_12202023_125957" --trained_guide_model_path "models/CartPole-v1_DQNConfig_12202023_125957/offline"

# For no offline dataset or pretrained guide
python run_full_training.py --env CartPole-v0 -dat pg -off bc -on dqn