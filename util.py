import csv
import os
import subprocess


def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)


def write_csv(data, fname):
    ensure_dir(fname)
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


def execute(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error