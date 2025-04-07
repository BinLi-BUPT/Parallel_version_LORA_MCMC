# Project Execution Documentation

## I. Project Setup

### 1. Cluster Configuration

This project is deployed across 20 servers, each equipped with AMD EPYC Bergamo processors operating at a base frequency of 3.1 GHz and 32 GiB of RAM. All servers run the 64-bit version of Ubuntu Server 24.04 LTS (Noble Numbat).

### 2. Environment Installation

To set up the project environment on each server, execute the following commands:

(1). Switch to the root user to gain administrative privileges

```bash
sudo su root
```

(2). Create a new directory called 'workspace'

```bash
mkdir workspace
```

(3). Give full read, write, and execute permissions to all users for the 'workspace' directory

```bash
chmod 777 -R workspace/
```

(4). Change the current working directory to 'workspace'

```bash
cd workspace/
```

(5). Download the Anaconda installer for Linux (version 2024.10-1)

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

(6). Run the Anaconda installation script

```bash
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

(7). Reload the shell configuration to activate Anaconda

```bash
source ~/.bashrc
```

(8). Use conda to install commonly used Python packages

```bash
conda install pip numpy scikit-learn scipy psutil joblib
```

(9). Use pip to install additional Python libraries not included in Anaconda

```bash
pip install ray tensorly
```

**Notes:**

- The Anaconda installer script (`Anaconda3-2024.10-1-Linux-x86_64.sh`) is downloaded from the official Anaconda repository.
- Ensure that the `workspace` directory has appropriate permissions to facilitate subsequent operations.
- The `conda` and `pip` commands install the necessary Python packages required for the project.

## II. Project Execution

### 1. Cluster Initialization

To initiate the Ray cluster:

- **On the head node**, execute:

  ```bash
  ray start --head --resources '{"node:RayNode1": 1}'
  ```

- **On each of the remaining servers**, execute:

  ```bash
  ray start --address='Your_IP_address:6379' --resources '{"node:RayNodeX": 1}'
  ```

  Replace `Your_IP_address` with the actual IP address of the head node and `RayNodeX` with a unique identifier for each node (e.g., `RayNode2`, `RayNode3`, ..., `RayNode20`).

**Notes:**

- The `--resources` flag assigns a custom resource label to each node, which can be useful for task scheduling within Ray.
- Ensure that the Ray version installed is compatible across all nodes to prevent potential conflicts.

### 2. Code Execution

If the program needs to be run **serially**：

```bash
python main.py
```

To monitor the execution results, view the log file:

```bash
cat log.txt
```

If you want to run programs in **parallel**：

On the head node, execute the main script:

```bash
python main_parallel.py
```

To monitor the execution results, view the log file:

```bash
cat log_parallel.txt
```

By following the above steps, the project environment will be correctly set up, and the Ray cluster will be properly initialized for distributed computation. 


**Notes:**
In parallel mode, when each process is performing the same task, due to factors such as operating system scheduling, resource contention (such as CPU, memory, and I/O), interference between processes, and other system overheads, the actual running time of each process may vary significantly. **Therefore, the execution time of the multi-process parallel execution part in this project comes from the process that executes the fastest among them.**
