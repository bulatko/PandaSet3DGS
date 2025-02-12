## Project structure:

```
PandaSet3DGS/
    - gaussian-splatting/
    - data/
        - base.py
        - pandaset.py
    - readme.md
```

## How to use:

```(python)
path_to_pandaset = '/path/to/pandaset'
path_to_output = '/path/to/output'

dataset = PandaSetDataset(path_to_pandaset)
scene = dataset[0]

scene.make_reconstruction()
scene.export(path_to_output)
```