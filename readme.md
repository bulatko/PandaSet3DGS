## Project structure:

```
PandaSet3DGS/
    - gaussian-splatting/
    - scripts/
        - converter.py
        - visualizer.py
    - convert.py
    - visualize.py
    - readme.md
```

## How to use:

```
python convert.py --input /path/to/pandaset/frames --output /path/to/processed
```

```
python gaussian-splatting/train.py -s /path/to/processed -o /path/to/output
```



