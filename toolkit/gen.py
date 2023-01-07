from subprocess import PIPE
import subprocess
from random import randint
from glob import glob

def set_seed(seed,file_name="generator/config.toml"):
    after = ""
    with open(file_name,"r") as file:
        for line in file:
            if line.startswith("seed"):
                after += f"seed = {seed}\n"
            else:
                after += line
    with open(file_name,"w") as file:
        print(after,file=file)

def gen_testcase():
    set_seed(randint(0,(1 << 64) - 1))
    subprocess.run("./generator/random_world.py -c ./generator/config.toml > generator/testcase.txt",shell=True)


def gen_configs():
    template = ""

    with open("generator/config.toml") as file:
        for line in file:
            if line[0] != "#":
                template += line

    #T_max = [300, 700, 1000]
    #map_max_depth = [5, 6, 7]
    #worker_num = [1, 2, 5, 10]
    #job_num = [250, 500, 1000]
    for i in [300, 700, 1000]:
        for j in [5, 6, 7]:
            for k in [1, 2, 5, 10]:
                for l in [250,500,1000]:
                    cur = template
                    cur += f"T_max = [{i}]\n"
                    cur += f"map_max_depth = [{j}]\n"
                    cur += f"worker_num = [{k}]\n"
                    cur += f"job_num = [{l}]\n"
                    print(cur,file=open(f"configs/config_{i}_{j}_{k}_{l}.toml","w"))


def gen_all():
    for i,config in enumerate(glob("configs/*")):
        if i != 3:
            continue
        set_seed(randint(0,(1 << 64) - 1),config)
        subprocess.run(f"./generator/random_world.py -c {config} > in/in{i:04d}.txt",shell=True)
        break


def main():
    # gen_testcase()
    gen_all()

if __name__ == '__main__':
    main()

