import sys
import os


class Logger(object):
    def __init__(self, target_file):
        # Remove old logging file on construction of logger
        try:
            if os.path.exists(target_file):
                os.remove(target_file)
        except OSError:
            print("Error while deleting file ", target_file)

        self.terminal = sys.stdout
        self.log = open(target_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
