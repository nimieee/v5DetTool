import multiprocessing
import subprocess
import sys


def run_app1():
    subprocess.run([sys.executable, "safety_aqm_online.py"])


def run_app2():
    subprocess.run([sys.executable, "safety_aqm_offline.py"])


def run_app3():
    subprocess.run([sys.executable, "safety_count.py"])


def run_app4():
    subprocess.run([sys.executable, "safety_process.py"])


def run_app5():
    subprocess.run([sys.executable, "safety_danger.py"])


def hello_world():
    print("Hello, World!")


def jet_best():
    print("jet_best")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    jet_best()
    # 创建两个进程分别运行两个应用
    # multiprocessing.Process(target=run_app1).start()
    multiprocessing.Process(target=run_app2).start()
    # multiprocessing.Process(target=run_app3).start()
    # multiprocessing.Process(target=run_app4).start()
    # multiprocessing.Process(target=run_app5).start()
