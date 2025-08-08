## Link to model weights and tokenizer ##
https://drive.google.com/drive/folders/1Hll-jU7r8HAlahEyjOtTOeKqZjikFPu4?usp=drive_link

## Installations ##

```
git clone https://github.com/aspuru-guzik-group/group-selfies.git

cd group-selfies
pip install .
cd ..

pip install -r requirements.txt
```

## Running ##

The function to run conditional molecule generation is `generate_molecules_from_properties()` in `conditional_generation.py`

The `model_path` argument is the path to the local directory where all of the model files (everything at the Google Drive link) are stored.