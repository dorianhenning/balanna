# Visualization Package for Human Bodys and Other Stuff

## File Reading

Features:
- [ ] `.pkl` reader for single frame data
- [ ] `.npz` reader for single frame data
- [ ] `.pkl` reader for multi frame data
- [ ] `.npz` reader for multi frame data
- [ ] `.json` reader for single frame data
- [ ] `.json` reader for multi frame data
- [ ] all file loading includes ground truth 6D-pose data
- [ ] all file loading uses estimated/supplied 6D-pose data

## SMPL Integration

Features:
- [ ] model loading (`torch` or `chumpy`/`numpy`?)
- [ ] fast return of mesh vertices & joints (batch computation for sequence)

## Mesh Visualization

Features:
- [ ] single frame (simple example scene)
- [ ] single frame comparison between `N` number of resulting meshes (e.g., optimization)
- [ ] sequence of single human mesh
- [ ] sequence of multiple humans (next to each other, separated by max distance)
- [ ] uncertainty coloring of mesh?

## Pointcloud Visualization

Features:
- [ ] single pointcloud of mesh
- [ ] uncertainty awareness/coloring of pointcloud

## Trajectory Plotting

- [ ] trajectory plotting over entire sequence
- [ ] trajectory plotting that is growing over sequence
- [ ] trajectory plot where the trajectory fades out (only last `N` frames)

## Image Keypoint Renderings

- [ ] render/projection of set of points in color
- [ ] projection of SMPL parameters directly
- [ ] projection of ,ultiple different estimates of
  - [ ] camera estimation
  - [ ] posture/shape estimation


## TODO
- visualization:
  - scene with markers (pointcloud)
  - moshed model
  - image (potentially synced)
