import numpy as np

from pathlib import Path


def main():
    directory = Path(__file__).parent.resolve() / "tmp"
    directory.mkdir(exist_ok=True)

    # Create txt file with numpy.
    points = np.random.rand(1000, 3) * 3 - 1.5
    colors = np.random.rand(1000, 3)
    with open(directory / "p_00.txt", 'w') as f:
        point_cloud = np.hstack((points, colors))
        np.savetxt(f, point_cloud)

    # Create txt file without numpy, e.g. from C++.
    points = np.random.rand(1000, 3) * 3 - 1.5
    with open(directory / "p_01.txt", 'w') as f:
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")


if __name__ == '__main__':
    main()
