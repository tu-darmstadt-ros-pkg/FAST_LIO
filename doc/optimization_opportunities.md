# FAST-LIO Optimization Opportunities

This document surveys performance bottlenecks and improvement candidates in the current
ROS2 codebase. Each item includes the relevant file and approximate line numbers, the
problem, and a concrete fix direction. Items are grouped roughly by risk and effort.

---

## Already Done

- **Multithreaded executor + callback groups** (`laserMapping.cpp`, `main()` and constructor):
  Subscription callbacks run on a `ReentrantCallbackGroup` so LiDAR/IMU messages are
  dequeued independently of the `MutuallyExclusiveCallbackGroup` that serializes the
  processing timers and services. Eliminates message-queue backlog during heavy frames.

- **Preprocessing outside the buffer lock** (item 1): `p_pre->process()` / `p_pre2->process()`
  now runs before `mtx_buffer.lock()` in all four LiDAR callbacks. Lock held only for buffer
  push. IMU callbacks no longer contend with scan preprocessing.

- **EKF early-exit** (item 2): Already present in `esekfom.hpp` (`if(t > 1 || i == maximum_iter - 1)`).
  No change needed.

- **Parallel L1/L2 undistortion** (item 3): `UndistortPclMultiLiDAR()` now uses
  `#pragma omp parallel sections num_threads(2)` for independent per-lidar backward
  propagation. `IMUpose` is read-only in both sections.

- **`IMUpose.reserve()`** (item 4): Both `UndistortPcl` and `UndistortPclMultiLiDAR` now
  call `IMUpose.reserve(v_imu.size() + 2)` before the forward-propagation loop, eliminating
  repeated vector reallocations.

- **Pre-allocated publish buffers** (item 5): `publish_frame_world`, `publish_frame_body`,
  `publish_effect_world`, and the per-frame world-cloud in `map_publish_callback` all use
  `static PointCloudXYZI::Ptr` that is `resize()`'d rather than heap-allocated each frame.

- **Static scratch buffers in `map_incremental`** (item 6): `PointToAdd` and
  `PointNoNeedDownsample` are now `static` local vectors that are `clear()`'d and
  conditionally `reserve()`'d, avoiding per-frame malloc/free.

- **TF extrinsic lookup off hot path** (item 7): Lookup moved from `timer_callback` to a
  dedicated `tf_init_timer_` (10 Hz, `processing_callback_group_`) that cancels itself once
  resolved. `timer_callback` now just checks the `l2_extrinsic_resolved` bool.

- **Intraprocess comms** (item 8): Default `NodeOptions` now sets
  `use_intra_process_comms(true)`. Zero-copy transfer is active automatically when the node
  runs inside a component container alongside its subscribers.

- **Async map publishing** (item 9): `map_publish_callback` does the synchronous
  accumulation step (transform current frame to world, append to `pcl_wait_pub`), then swaps
  the cloud out and dispatches voxel-filter + `toROSMsg` + publish to a `std::async` worker.
  The `processing_callback_group_` is unblocked for the next timer / diagnostics tick while
  the heavy serialisation runs on a separate thread. A `std::future` guard prevents
  concurrent publish tasks.

---

## High Impact, Low Risk

### 1. Move preprocessing outside the buffer lock

**File:** `src/laserMapping.cpp` — `standard_pcl_cbk` (~line 1475), `livox_pcl_cbk` (~line 1489)

`p_pre->process()` is called while `mtx_buffer` is held. Preprocessing a full scan (feature
extraction, outlier removal, point parsing) can take several milliseconds. During that window
the IMU callback blocks on the same mutex, causing timestamp jitter in the IMU buffer.

**Fix:** allocate and preprocess the cloud before taking the lock; only push the result
into the buffer while locked.

```cpp
PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
p_pre->process(msg, ptr);          // outside lock
double cur_time = get_time_sec(msg->header.stamp);
{
    std::lock_guard<std::mutex> lk(mtx_buffer);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
}
sig_buffer.notify_all();
```

The only shared state read before the lock (`last_timestamp_lidar`, `is_first_lidar`) would
need a separate small lock or atomic — both are cheap reads.

---

### 2. Early-exit from the iterated EKF update

**File:** `include/IKFoM_toolkit/esekfom/esekfom.hpp` — `update_iterated_dyn_share_modified()`

The loop runs a fixed `NUM_MAX_ITERATIONS` (default 4). In practice the state often converges
after 2–3 iterations; the remaining iterations re-run `h_share_model` (KDTree search +
Jacobian build) for no gain.

**Fix:** add a consecutive-convergence counter: if the state increment satisfies the
convergence test twice in a row, break early. This is already partially implemented via the
`flg_EKF_converged` flag in `laserMapping.cpp` but the break is not fed back into the loop
inside `esekfom.hpp`.

Estimated saving: 15–25 % of `timer_callback` runtime on well-conditioned frames.

---

### 3. Parallel L1/L2 undistortion in bundle mode

**File:** `src/IMU_Processing.hpp` — `UndistortPclMultiLiDAR()` (~line 434)

After the shared IMU forward propagation, L1 and L2 undistortion passes are independent of
each other. They are currently sequential.

**Fix:**

```cpp
#pragma omp parallel sections num_threads(2)
{
    #pragma omp section
    { /* undistort pcl_L1 */ }
    #pragma omp section
    { /* undistort pcl_L2 */ }
}
```

OpenMP is already pulled in (`#include <omp.h>`, line 35 of `laserMapping.cpp`) and used in
`h_share_model`. No new dependency needed.

Estimated saving: 40–50 % of undistortion time in dual-lidar bundle mode.

---

### 4. `IMUpose` pre-allocation

**File:** `src/IMU_Processing.hpp` — `ImuProcess::Process()` (~line 499)

`IMUpose` is a `vector<Pose6D>` that grows via `push_back` once per IMU message during
forward propagation. IMU arrives at 100–400 Hz; a 100 ms LiDAR scan window holds 10–40
messages. No `reserve()` is called, so the vector reallocates 4–5 times per frame.

**Fix:** at the top of `Process()`, call `IMUpose.reserve(v_imu.size() + 2)` after clearing.

---

## Medium Impact, Moderate Effort

### 5. Avoid double copy on point cloud publish

**File:** `src/laserMapping.cpp` — `publish_frame_world()` (~line 704)

The current path:
1. Allocates a new `PointCloudXYZI` (one copy).
2. Calls `pcl::toROSMsg()` which serialises into a `sensor_msgs::PointCloud2` (second copy).

**Fix:** Build the `PointCloud2` message directly, writing the transformed points into
`msg->data` without the intermediate PCL cloud. This halves the memory traffic for large
clouds.

For downstream consumers in the same process, enabling `use_intraprocess_comms` in the node
options eliminates serialisation entirely (see item 8).

---

### 6. Reuse per-frame scratch buffers in `map_incremental`

**File:** `src/laserMapping.cpp` — `map_incremental()` (~line 653)

```cpp
PointVector PointToAdd;
PointVector PointNoNeedDownsample;
PointToAdd.reserve(feats_down_size);
PointNoNeedDownsample.reserve(feats_down_size);
```

These vectors are allocated and freed every frame. `feats_down_size` fluctuates but is
typically 2 000–8 000 points. Promoting them to class members (or `static`) lets the OS
re-use already-mapped pages and avoids malloc/free churn.

---

### 7. Move the TF-based L2 extrinsic lookup out of `timer_callback`

**File:** `src/laserMapping.cpp` — `timer_callback()` (~line 1607)

While `l2_extrinsic_resolved` is false, every timer tick calls `tf_buffer_->lookupTransform()`
which can block 5–50 ms. This directly degrades the 10 Hz processing loop during startup.

**Fix:** Move the TF lookup to a one-shot `rclcpp::TimerBase` that fires at 10 Hz, stops
itself once the transform is found, and sets `l2_extrinsic_resolved`. The main
`timer_callback` then skips the branch entirely after init.

---

### 8. Enable intraprocess communication

**File:** `src/laserMapping.cpp` — node constructor / `main()`

When the node runs inside a component container alongside the LiDAR driver, all published
topics (odometry, point clouds, path) pass through ROS2 serialisation even though publisher
and subscriber live in the same process. The commit that introduced the component registration
(`RCLCPP_COMPONENTS_REGISTER_NODE`) already set the stage.

**Fix:** pass `rclcpp::NodeOptions().use_intraprocess_comms(true)` from the container launch
file (or component manager). No code change required in the node itself; the container
orchestrates it. Zero-copy transfer is then used for `unique_ptr`-published messages.

Relevant to outbound odometry and path topics when the navigation stack runs in the same
container.

---

### 9. Asynchronous map publishing

**File:** `src/laserMapping.cpp` — `map_publish_callback()` (~line 844)

When `map_pub_en` is true this callback iterates over the full KDTree (potentially 100 k+
points), optionally runs a VoxelGrid filter, then serialises to a `PointCloud2`. This runs
on the `processing_callback_group_` and therefore blocks the diagnostics and main processing
timers for its duration (can be 100–200 ms).

**Fix:** run the map serialisation inside a `std::async` task or a dedicated
`rclcpp::CallbackGroup` with its own thread. The KDTree snapshot can be taken under a brief
lock; serialisation then happens off the hot path.

---

## Lower Impact / Algorithmic Scope

### 10. Adaptive VoxelGrid leaf size

**File:** `src/laserMapping.cpp` — downsampling (~line 1670)

`filter_size_surf_min` is a fixed config parameter. In close-range environments the scan is
dense and a larger leaf wastes less information; at long range a smaller leaf helps coverage.
An adaptive leaf size keyed on the median point-to-point distance of the current scan could
maintain a more uniform information budget across scenarios.

---

### 11. KDTree rebuild threshold as a ROS parameter

**File:** `include/ikd-Tree/ikd_Tree.h` — line 14–18 (`#define` constants)

`Multi_Thread_Rebuild_Point_Num = 1500` and `ForceRebuildPercentage = 0.2` are compile-time
constants. High-density environments trigger more frequent forced rebuilds; sparse
environments may never rebuild. Exposing these as ROS parameters (or at least as
`cmake` options) allows tuning per platform without recompiling.

---

### 12. Nearest-neighbour count as a ROS parameter

**File:** `include/common_lib.h` — `#define NUM_MATCH_POINTS 5`

This controls how many surface neighbours are searched per feature point during the EKF
update. A value of 5 is conservative; fewer neighbours reduce KDTree search time at the cost
of slightly noisier plane estimates. Making this a runtime parameter allows profiling the
accuracy/speed trade-off per sensor.

---

### 13. Pre-filter point cloud before undistortion

Currently the full raw cloud (~10 k–30 k points) is undistorted, then downsampled. Running
a coarse voxel filter (e.g., 1 cm leaf) immediately after preprocessing and before
`UndistortPcl` reduces the number of points that go through the per-point IMU interpolation
loop, at the cost of slightly less accurate distortion correction on the filtered-out points.
Worth benchmarking on the target sensor.

---

## Summary Table

| # | Area | File | Risk | Effort | Impact |
|---|------|------|------|--------|--------|
| 1 | Preprocessing outside lock | `laserMapping.cpp` | Low | Low | High |
| 2 | EKF early-exit | `esekfom.hpp` | Low | Low | High |
| 3 | Parallel L1/L2 undistort | `IMU_Processing.hpp` | Low | Low | High (dual-lidar) |
| 4 | `IMUpose.reserve()` | `IMU_Processing.hpp` | Low | Trivial | Medium |
| 5 | Single-copy publish | `laserMapping.cpp` | Low | Medium | Medium |
| 6 | Static scratch buffers | `laserMapping.cpp` | Low | Trivial | Low–Medium |
| 7 | TF lookup off hot path | `laserMapping.cpp` | Low | Medium | Medium |
| 8 | Intraprocess comms | launch config | Low | Low | Medium |
| 9 | Async map publish | `laserMapping.cpp` | Medium | Medium | Medium |
| 10 | Adaptive downsampling | `laserMapping.cpp` | Medium | Medium | Low–Medium |
| 11 | KDTree params as ROS params | `ikd_Tree.h` | Low | Low | Low–Medium |
| 12 | `NUM_MATCH_POINTS` as param | `common_lib.h` | Low | Low | Low |
| 13 | Pre-filter before undistort | `laserMapping.cpp` / `IMU_Processing.hpp` | Medium | Low | Low–Medium |
