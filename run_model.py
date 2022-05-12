from unityagents import UnityEnvironment

from tennis import Environment, plot_scores


def run_model():
    unity_env_file = 'Tennis_Linux/Tennis.x86_64'
    env = Environment(UnityEnvironment(file_name=unity_env_file))
    scores = env.run_model(['actor_0.pth','actor_1.pth'])
    env.close()


if __name__ == "__main__":
    run_model()
