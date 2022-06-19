import argparse
import os
import datetime

parser = argparse.ArgumentParser(description="A script for counting number of code lines.")
parser.add_argument("-d", "--directory", default="..", help="The directory for counting code lines.")
args = parser.parse_args()

def countLines(root: str, filename: str):
    """Counts lines in the given file."""
    if not filename.endswith(('.py', '.h', '.hpp', '.cpp', 'CMakeLists.txt')):
        return 0
    with open(os.path.join(root, filename), 'r', encoding='utf8') as f:
        count = len(f.readlines())
        print(os.path.join(root, filename), count)
    return count

lines = 0
for item in os.scandir(args.directory):
    if item.name in (".vscode", "build", "dependency", "image"):
        continue
    if item.is_dir():
        for root, dirname, filesname in os.walk(os.path.join(args.directory, item.name)):
            for filename in filesname:
                lines += countLines(root, filename)
    elif item.is_file():
        lines += countLines(args.directory, item.name)

print('\nTotal lines: {}, time is {}'.format(lines, datetime.datetime.now()))
