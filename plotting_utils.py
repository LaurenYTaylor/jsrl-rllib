import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

online_data = pd.read_csv("data/CartPole-v1_DQNConfig_12212023_113551_online.csv")
"""
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(13,6))

guide_policy_data = np.load("data/guide_policy_res.npy")
ax[0].plot(guide_policy_data, label="Post-Training Guide Policy")
ax[0].set_xlabel("Post-Training Evaluation Iteration")
ax[0].set_ylabel("Reward")
ax[0].legend()


ax[1].plot(online_data['Step'], online_data['Value'], label="Online Training with Guide Policy")
ax[1].set_xlabel("Training Step")
guide_policy = pd.read_csv("data/run-CartPole-v1_DQNConfig_12212023_113551_online-tag-ray_tune_info_learner_jsrl_current_horizon.csv")
ax[1].plot(guide_policy['Step'], guide_policy['Value'], label="Horizon")
ax[1].legend() 

plt.savefig("plots/offline_to_online.png", dpi=300)
plt.show()
plt.close()
"""
no_guide = pd.read_csv("data/CartPole-v1_DQNConfig_12212023_120002_noguide.csv")
plt.plot(
    online_data["Step"], online_data["Value"], label="Online Training with Guide Policy"
)
plt.plot(
    no_guide["Step"], no_guide["Value"], label="Online Training without Guide Policy"
)
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.legend()

plt.savefig("plots/guide_vs_no_guide.png", dpi=300)
plt.show()
