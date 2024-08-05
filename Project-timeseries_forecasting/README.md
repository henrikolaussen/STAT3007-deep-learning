# STAT3007-timeseries_forecasting

# Usage:
Install poetry from here:
[official Poetry documentation](https://python-poetry.org/docs/#installing-with-the-official-installer).


# Install dependencies and the project as a package with poetry 
```bash
poetry install
```



HOW TO RANGPUR:

create a conda environment for python 3.10.12

activate said environment
run pip install -e . when you are located at the root of this project.

Additionally you might need to run pip install torch==2.0.1+cu118


Lastly run sbatch rangpur.sh


## How to move weights back from the cluster onto local machine

### For a folder
```bash
scp -r <student_number>@rangpur.compute.eait.uq.edu.au:<path_to_folder_on_cluster> <local_folder>
```

### For a single file
```bash
scp <student_number>@rangpur.compute.eait.uq.edu.au:<path_to_file_on_cluster> <local_folder>
```
