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

```
python convert.py --input /path/to/pandaset/frames --output /path/to/processed
```

```
python gaussian-splatting/train.py -s /path/to/processed -o /path/to/output
```



