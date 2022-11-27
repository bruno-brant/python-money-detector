# Modules

## money_counter
Definitions for creating and training the Money Detection network. 

# server
A flask server to expose an API that runs the Money Detection model.

# Development

# Jupyter Notebooks

## Enviroment variables

Set the following environment variables to configure the notebooks:

| Name              | Value                                           |
| ----------------- | ----------------------------------------------- |
| COINS_DATASET_DIR | The path to the coins.json file of the dataset. |

## Configuring sys.path

Notebooks must resolve modules that are in the repository root and, for that, one
must set the `sys.path` variable. To do that without poluting the notebook 
with local environment data, in *VSCode*, one can use the setting `jupyter.runStartupCommands` 
adding the necessary code to configure it:

```json
 "jupyter.runStartupCommands": [
        "import sys",
        "sys.path.append('..\\') # Relative to the notebook, not the workspace",
    ],
```
   