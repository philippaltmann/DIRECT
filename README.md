# DIRECT 
Discriminative Reward Co-Training

## Setup 

### Requirements
- python 3.8.6
- cuda 11.3 or 10.2

### Installation 
```sh
$ pip install -r requirements.txt
```

## Fix Warnings in used packages 

`/.direnv/python-3.7.7/lib/python3.7/site-packages/pycolab/ascii_art.py:318: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.`

change ascii_art.py line 318 from 
`art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in art)`
to 
`art = np.vstack([np.fromstring(line, dtype=np.uint8) for line in art])`