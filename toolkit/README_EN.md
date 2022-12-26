# Contents
- Generator for test cases
- Judge program
- Sample codes
- Visualizer

# Execution Environment
The following tools are required:
- C++ compiler that supports C++17
- Python(Python 3.8 or later)
  - The packages listed in `requirements.txt` must be able to be installed using `pip` (See **Preparation**)
- bash(Tested in 5.0.17(1))

## Examples of execution enviroment

### Cygwin(3.4.2-1, setup ver.:2.923)
With the following packages:
- gcc-g++ (Tested in 10.2.0-1)
- python38-devel(Tested in 3.8.9-2)
- python38-numpy(Tested in 1.21.4-1)

### Ubuntu 20.04(ubuntu:20.04 on Docker)
With the following packages:
- `g++` (Tested in 4:9.3.0-1ubuntu2 amd64)
- `python3-pip` (Tested in 20.0.2-5ubuntu1.6 all)
- `python3-numpy` (Tested in 1:1.17.4-5ubuntu3.1 amd64)
```bash
sudo apt install g++ python3-pip python3-numpy
```

### Ubuntu 22.04(ubuntu:22.04 on Docker)
With the following packages:
- `g++` (Tested in 4:11.2.0-1ubuntu1 amd64)
- `python3-pip` (Tested in 22.0.2+dfsg-1 all)
- `python3-numpy` (Tested in 1:1.21.5-1ubuntu22.04.1 amd64)
```bash
sudo apt install g++ python3-pip python3-numpy
```


# Preparation
1. `./build.sh`
2. `python3 -m pip install -r requirements.txt`

# Execution

## Generating test cases
```bash
cd generator

# Generate config.toml, a configuration file (with default parameters)
./random_world.py -g config.toml 

# If needed, change values of the items in config.toml. Each item has a description.
# To generate a test case for Task A, set the value of 'type' to "A" (type = "A"). "B" for Task B.

# Read the configuration file and generate testcase.txt, a test case file for the task type specified in config.toml
./random_world.py -c config.toml > testcase.txt 
```

## Executing judge program
To run the judge program, give a command to run your answer code as an argument of `judge.sh`:
```bash
./judge.sh <test case file path> <log file output path> <command to run your answer code...>
```

Example 1 (create an executable (`a.out` for instance) and run it):
```bash
./judge.sh generator/testcase.txt visualizer/default.json ./a.out
```

Example 2 (run a python code(`answer.py` for instance)):
```bash
./judge.sh generator/testcase.txt visualizer/default.json python3 answer.py
```

Example 3 (ignore all inputs and output a text file (`answer.txt` for instance)):
```bash
./judge.sh generator/testcase.txt visualizer/default.json sh -c "cat > /dev/null|cat answer.txt"
```

## Running sample codes

For Task A:
```bash
./judge.sh generator/testcase.txt visualizer/default.json sample/sample_A
```
(Generate `generator/testcase.txt` for Task A beforehand.)

For Task B'
```bash
./judge.sh generator/testcase.txt visualizer/default.json sample/sample_B
```
(Generate `generator/testcase.txt` for Task B beforehand.)

## Visualizer (`visualizer/index.html?en`)

See `visualizer/index.html?en`. A log file (output specified in the second argument of `judge.sh`) is used to visualize the result. The query `?en` in the url enables English descriptions.
