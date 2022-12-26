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
