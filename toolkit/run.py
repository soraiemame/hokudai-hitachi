from subprocess import PIPE
import subprocess
from glob import glob


def once():
    print("gen")
    subprocess.run("python3 gen.py",shell=True,stdout=PIPE,stderr=PIPE)
    print("run")
    proc = subprocess.run("./judge.sh generator/testcase.txt visualizer/default.json cmd.exe /c cargo run --release",shell=True,stdout=PIPE,stderr=PIPE,encoding="utf-8")
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
    print("clear...")
    subprocess.run("rm -rf out",shell=True)
    subprocess.run("mkdir out",shell=True)
    subprocess.run("cmd.exe /c cargo build --release",shell=True,stdout=PIPE,stderr=PIPE)
    runc = "cmd.exe /c ..\\\\target\\\\release\\\\hokudai-hitachi.exe"
    print("Compiling code...")
    subprocess
    cnt = 0
    error = False
    for (i,test) in enumerate(glob("in/*")):
        proc = subprocess.run(f"./judge.sh {test} out/out{i:04d}.json {runc}",shell=True,stdout=PIPE,stderr=PIPE,encoding="utf-8")
        if proc.returncode != 0:
            print(f"Testcase {i}({test}): ERROR")
            error = True
            break
        else:
            score = proc.stderr.split()[-1]
            score = int(score.replace("score:",""))
            print(f"Testcase {i}({test}) score: {score:,}")
            print(f"Testcase {i}({test}) score: {score:,}",file=open("out/scores.txt","a"))
            cnt += score
    if error:
        print("Error occured.")
    else:
        print("All testcases passed.")
        print(f"score sum: {cnt:,}")
        print(f"testcase average: {int(cnt / 108):,}")
        print(f"50 testcase conversion: {int(cnt / 108 * 50):,}")

def main():
    # once()
    run_108()

if __name__ == '__main__':
    main()
