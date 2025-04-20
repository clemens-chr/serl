import wandb
import subprocess
import time
import yaml

def main():
    run = wandb.init()
    config = run.config

    # Dump config to YAML for the scripts
    with open("sweep_config.yaml", "w") as f:
        yaml.dump(dict(config), f)

    learner = subprocess.Popen(["bash", "learner.sh", "sweep_config.yaml"])
    actor = subprocess.Popen(["bash", "actor.sh", "sweep_config.yaml"])

    time.sleep(3600)
    learner.terminate()
    actor.terminate()

    run.finish()

if __name__ == "__main__":
    main()
