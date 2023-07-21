# Natural language to code language - Analysing Sofware using Deep Learning


## Requirements

- Python >= 3.8
- transformers
- torch
- numpy>=1.20.0
- scikit-learn


## #1 Clone Repository

\*_Note: Private repo._

```{bash}
git clone https://github.com/senatirpan/ASDL.git
cd ASDL
```

## #2 Setup Python Environment

From inside the repository folder run the following commands.

```{python}
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## #3 Install and split data

Dataset available here, download "train.json" and "dev.json" into train_data folder: (https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code/dataset/concode)
Then run all split_data.ipynb, "data" folder created automatically.

## #4 Run encoder_decoder.ipynb

This jupyter notebook contains tokenization, translation and training parts, respectively.

