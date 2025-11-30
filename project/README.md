# Air-Quality-G1
This is the repository for a Machine Learning project.

## Introduction
The task is to predict the PM2.5 concentration based on various parameters, such as dew point or temperature, and classify whether it is at a healthy level. It matters because PM2.5 is associated with various heart and lung diseases. Our goal is to determine whether an air environment is healthy for human lives.​

## Dataset
[Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data), licensed under a CC-BY 4.0 license.

## How to use
**Prerequisites:** 
```
git
python
pip
```
**Step 1:** Clone the repository
```
git clone https://github.com/HaisenW/Air-Quality-G1.git
cd Air-Quality-G1
```
**Step 2:** Set up the appropriate Python environment
```
python -m venv cs4120-env
source cs4120-env/bin/activate
pip install -r project/requirements.txt
```
The repository is now ready for use.

## AI Usage Disclosure Statement
The following AI models were used for idea generation, concept clarification, and basic troubleshooting:
- Alibaba Qwen3-235B
- OpenAI GPT5
- Z.ai GLM-4.6

Here is an example of how they were used:
```
Query: How to make pip replicate the environment in a requirements.txt file?

Response: To replicate a Python environment using a requirements.txt file with pip, follow these steps:

Generate the requirements file: In the source environment, use the command
pip freeze > requirements.txt
This creates a requirements.txt file listing all installed packages and their exact versions. 

Install packages in the target environment: On the destination machine or in a new virtual environment, run
pip install -r requirements.txt
This installs all the packages listed in the file, replicating the original environment. 

Use virtual environments: It is strongly recommended to use a virtual environment (e.g., created with python -m venv myenv) to avoid conflicts with system-wide packages and ensure isolation. Activate it before installing the requirements. 

This method ensures reproducibility of Python environments across different systems.
```
