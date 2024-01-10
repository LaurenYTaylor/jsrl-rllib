training_configs = {
    "cql": {
        "training": {
            "lagrangian": True,
            "min_q_weight": 5.0,
            "lagrangian_thresh": 5.0,
            "num_actions": 10,
            "bc_iters": 20000,
            "temperature": 1.0,
            "q_model_config": {"fcnet_hiddens": [256, 256, 256]},
            "policy_model_config": {"fcnet_hiddens": [256, 256, 256]},
            "optimization_config": {
                "actor_learning_rate": 3e-5,
                "critic_learning_rate": 3e-4,
                "entropy_learning_rate": 1e-4,
            },
        },
        "rollouts": {
            "rollout_fragment_length": "auto",
        },
        "offline_data": {"actions_in_input_normalized": True},
    },
    "bc": {"training": {}, "rollouts": {}, "offline_data": {}},
}
