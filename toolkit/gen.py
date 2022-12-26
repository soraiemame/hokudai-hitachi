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


def gen_all():
    for i,config in enumerate(glob("configs/*")):
        set_seed(randint(0,(1 << 64) - 1),config)
        subprocess.run(f"./generator/random_world.py -c {config} > in/in{i:04d}.txt",shell=True)


def main():
    # gen_testcase()
    gen_all()

if __name__ == '__main__':
    main()

