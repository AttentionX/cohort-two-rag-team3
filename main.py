import argparse
from agent import generate_specific_tasks

def main(goal: str):
    generate_specific_tasks(goal=goal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate specific tasks.')
    parser.add_argument('goal', type=str, help='The goal to generate tasks for.')
    args = parser.parse_args()
    main(args.goal)