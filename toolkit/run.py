from subprocess import PIPE
import subprocess
from sys import argv
import time

def once():
    print("gen")
    subprocess.run("python3 gen.py",shell=True,stdout=PIPE,stderr=PIPE)
    print("run")
    proc = subprocess.run("./judge.sh generator/testcase.txt visualizer/default.json cmd.exe /c cargo run --release --bin a",shell=True,stdout=PIPE,stderr=PIPE,encoding="utf-8")
    if proc.returncode != 0:
        return False
    else:
        print(proc.stderr.split()[-1])
        return True

def mul():
    for i in range(100):
        ok = once()
        if not ok:
            break

def run_108():
    problem = argv[1].lower()
    if not problem in "ab":
        print(f"Invalid problem: {problem}")
        exit(1)
    print("clear...")
    subprocess.run("rm -rf out",shell=True)
    subprocess.run("mkdir out",shell=True)
    if len(argv) == 2:
        print("Compiling code...")
        cc = f"cmd.exe /c cargo build --release --bin {problem}"
        subprocess.run(cc,shell=True)
    runc = f"cmd.exe /c ..\\\\target\\\\release\\\\{problem}.exe" if len(argv) == 2 else f"../target/release/{problem}"
    cnt = 0
    for i in range(108):
        try:
            test = f"in/in{i:04d}.txt"
            proc = subprocess.run(f"./judge.sh {test} out/out{i:04d}.json {runc}",shell=True,stdout=PIPE,stderr=PIPE,encoding="utf-8")
            if proc.returncode != 0 and not proc.stderr.split()[-1].startswith("score"):
                print(f"Testcase {i}({test}): ERROR")
                print(proc.stderr)
                exit(1)
            else:
                score = proc.stderr.split()[-1]
                score = int(float(score.replace("score:","")))
                spaces = " " * (4 - len(str(i)))
                print(f"Testcase {i}({test}){spaces}score: {score:,}")
                print(f"Testcase {i}({test}){spaces}score: {score:,}",file=open("out/scores.txt","a"))
                cnt += score
        except KeyboardInterrupt:
            time.sleep(5)
            exit(1)
    print("All testcases passed.")
    print(f"score sum: {cnt:,}")
    print(f"testcase average: {int(cnt / 108):,}")
    print(f"50 testcase conversion: {int(cnt / 108 * 50):,}")



def main():
    if len(argv) == 1:
        print("Please specify problem.")
        exit(1)
    # once()
    run_108()

if __name__ == '__main__':
    main()
