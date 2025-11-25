## Create a new virtual env in .venv
```bash
python3 -m venv .venv
```

## Activate venv
```bash
source .venv/bin/activate
```
## Install cyclonedds
Follow the instructions in the [Cyclone DDS repository](https://cyclonedds.io/docs/cyclonedds/latest/installation/installation.html).

> [!IMPORTANT]  
> Make sure that cyclonedds core 0.10.2 is installed (e.g. git checkout 0.10.2)

## Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Install the Unitree Go2 SDK
Follow the instructions in the [Unitree Go2 SDK repository](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#installing-from-source).

## Run the program
```bash
python src/main.py
```