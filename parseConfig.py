import argparse
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-device", default="0")
parser.add_argument("-gamma", type=float, default=0.99)
parser.add_argument("-epsilon", type=float, default=0.05)
parser.add_argument("-buff_size", type=float, default=4)
parser.add_argument("-load_checkpoint", default="")
parser.add_argument("-agent", default="DQN")
parser.add_argument("-env_name", default="Breakout-v0")
parser.add_argument("-num_episodes", type=int, default=100000)
config = parser.parse_args()
config.logging = config.logging not in ["0", "false", "False"]
config.device = "/gpu:"+config.device

print("Logging: " + str(config.logging))
if config.transition_function not in [
        "oh_concat", "expanded_concat", "conditional"]:
    raise Exception(config.transition_function+" is not valid transition function")

if config.logging:
    # find the number of this run
    int_folders = []
    for folder in os.listdir("log"):
        folder_num = re.search(r"\d+", folder)
        if folder_num:
            int_folders.append(int(folder_num.group()))
    config.run_name = str(max(int_folders + [0]) + 1)+"-"+config.agent+"-"+config.env_name
    config.log_path = "log/" + config.run_name + "/"

    config.checkpoint_path = config.log_path + "checkpoint/"
    os.makedirs(config.checkpoint_path)

    # log all the config
    config_log_file = open(config.log_path + "config.txt", 'w+')
    config_vars_dict = vars(config)
    for var in config_vars_dict:
        config_log_file.write(var + ": " + str(config_vars_dict[var]) + "\n")
    config_log_file.close()
