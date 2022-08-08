import argparse

def parseArgument() -> argparse.Namespace:
    description = "This file is the entrance of training nerf. Try to get help by using -h."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", default='', help="The path to the configuration file for training nerf.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="The gpu id for training nerf.")
    parser.add_argument("-r", "--resume", type=int, default=0, help="Train from the specified iteration, 0 for starting from the scratch, -1 for starting from the maximum iteration that is saved.")

    args = parser.parse_args()
    return args
