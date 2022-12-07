# Map Gaze to Surface Without Delay

This script demonstrates how to map gaze to a surface based on the last known surface
location. This reduced the mapping delay with the disadvantage that one does not know
if the last known surface definition is still accurate, e.g. there is way to tell
whether the surface has disappeared.

## Installation

Requires Python 3.7 or newer

```
python -m pip install -r requirements.txt
```

## Usage

```
python map-gaze-to-surface-no-delay.py [-h] [-a ADDRESS] [-p PORT] [-n NAME]

optional arguments:
  -h, --help            show this help message and exit
  -a ADDRESS, --address ADDRESS
  -p PORT, --port PORT
  -n NAME, --name NAME
```

(Optional parameters are given in `[...]`)

**Caveat**: This script assumes the default scene camera resolution (1280x720). The
corresponding intrinsics need to be changed if a different resolution or camera lens
is being used.
