# Balanna: Easy 2D & 3D Visualizations

## Installation
```
pip install balanna
```

## Minimal Example
```
import numpy as np

from balanna.trimesh import show_point_cloud
from balanna.display_scenes import display_scenes


def main():
    pcs = np.random.rand((20, 3))
    for t in range(20):
        scene = show_point_cloud(pcs[t])
        yield {'point_cloud': scene}


if __name__ == '__main__':
    display_scenes(main())
```

## Visualization of cached directory
```
python3 -m balanna <cached-directory>
```
