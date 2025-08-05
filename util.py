import csv
import subprocess
import os
from pathlib import Path

uid = os.getuid()
gid = os.getgid()

def write_csv(data, fname):
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


def execute(cmd, volume_mount: Path = None, use_docker: bool = False):
    if not use_docker:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error
    else:
        import docker
        client = docker.from_env()
        print(f"Running command in Docker: {cmd}")
        container = client.containers.run(
            "rost-cli:latest",
            cmd,
            volumes={volume_mount: {'bind': volume_mount.as_posix(), 'mode': 'rw'}},
            user=f"{uid}:{gid}",
            detach=True
        )
        container.wait()
        output = container.logs()
        container.remove()
        return output, ""


def map_range(x, a, b, c=0, d=1):
    y = (x-a) * (d-c)/(b-a) + c
    return y
