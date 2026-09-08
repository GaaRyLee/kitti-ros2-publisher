# DrivingStereo mapping demo notes

## What the demo demonstrates

The demo creates an inspectable sparse mapping baseline from DrivingStereo:

1. Stereo images are replayed as synchronized ROS 2 topics.
2. PSMNet produces disparity, which is back-projected into pseudo-points.
3. The learned filter removes low-confidence pseudo-points.
4. ORB features from consecutive left images are matched.
5. Stereo depth at prior-frame ORB features forms 3D–2D correspondences.
6. OpenCV PnP-RANSAC estimates the relative camera pose.
7. Each filtered point cloud is colorized from the left image, transformed once
   into a fixed `map` frame, voxel-downsampled, and published to RViz2.

The map uses the ROS ground convention `X-forward, Y-left, Z-up`. Input
camera coordinates are OpenCV/ROS optical `X-right, Y-down, Z-forward`; the
initial optical-to-map rotation is applied before poses are accumulated.

## Why the transform does not repeatedly move old points

For an estimated transform `T_current_previous`, the implementation updates:

```text
T_map_current = T_map_previous * inverse(T_current_previous)
```

Only points from the newly processed camera frame are transformed by
`T_map_current`. Existing map points are already in `map` and are not
transformed again. This prevents the common bug of repeatedly translating an
entire accumulated map in camera coordinates.

## Pose diagnostics

`output/pseudo_points/pose_log.csv` records:

- source frame index and timestamp;
- accepted/rejected PnP result;
- ORB ratio-match count and PnP-RANSAC inlier count;
- relative translation and rotation;
- accumulated camera position in `map`.

Interpret pose by physical time, not by a fixed per-frame distance. DrivingStereo
timestamps are irregular, so a multi-metre motion across a longer interval can
still be plausible. Slow replay until frame indices processed by the mapper are
near-consecutive; PnP is intended for temporally close image pairs.

## Current expected behavior

- Road markings, trees, and other static geometry should substantially overlap.
- Moving vehicles remain in the static map and produce visible ghosting. This is
  expected for a semantic-free mapper.
- Far-range pseudo-depth is less stable than near and mid-range depth, so small
  drift may still be visible in a bird's-eye view.

## Natural follow-up work

1. Add time-aware pose gates based on translation speed, yaw rate, and inlier
   quality.
2. Refine accepted PnP-RANSAC estimates with iterative/LM optimization.
3. Mask dynamic classes before map accumulation.
4. Add local ICP refinement after a trusted PnP initialization.
5. Add loop closure with RTAB-Map or migrate to an ORB-SLAM-based backend.
