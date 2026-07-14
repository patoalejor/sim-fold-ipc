# Implementation Plan: C-IPC Cosserat Rod in NVIDIA Warp

## 1. Project Overview
This document outlines the step-by-step implementation of a **Codimensional Incremental Potential Contact (C-IPC)** solver for a Cosserat rod within **NVIDIA Warp**. This implementation is designed to simulate endovascular guidewire navigation inside a static 15k-triangle vascular mesh. 

This repository will serve as a direct benchmark against the existing **XPBD (Extended Position-Based Dynamics)** implementation, focusing on physical accuracy (zero self-intersection/tunneling) vs. computational throughput.

---

## 2. Architecture & Data Structures

Unlike XPBD which projects constraints directly, C-IPC formulates each time step as an energy minimization problem. You will use Warp's autograd capabilities or write custom gradient/Hessian kernels.

### 2.1 State Representation (Warp Arrays)
A Cosserat rod requires tracking both position and orientation (material frames) for each node.
* `x`: `wp.array(dtype=wp.vec3)` (Nodal positions)
* `q`: `wp.array(dtype=wp.quat)` (Nodal orientations / material directors)
* `v`, `omega`: `wp.array(dtype=wp.vec3)` (Linear and angular velocities)
* `x_prev`, `q_prev`: State at the previous timestep.

### 2.2 Static Obstacle (Vessel Mesh)
* `vessel_verts`: `wp.array(dtype=wp.vec3)`
* `vessel_indices`: `wp.array(dtype=int)`
* **Broad-phase Structure:** `wp.HashGrid` or `wp.Mesh` (Warp’s built-in BVH for mesh distance queries).

---

## 3. Step-by-Step Implementation

### Phase 1: Cosserat Elasticity Kernels
Define the internal strain energy $E_{elastic}$ of the guidewire.
1. **Stretch & Shear Energy:** Compute the difference between the tangent vector and the director frame.
2. **Bending & Twisting Energy:** Compute the Darboux vector from the relative rotation between adjacent quaternions.
3. **Implementation:** Write a `@wp.kernel` that takes `x` and `q`, computes the total elastic energy, and (optionally) uses Warp's `wp.expect_ad()` for automatic differentiation of gradients.

### Phase 2: The IPC Barrier Potential
This is the core differentiator from XPBD. The barrier ensures the wire never clips the vessel.
1. **Distance Queries:** Use Warp's `wp.mesh_query_point` and `wp.mesh_query_edge` to find the closest points on the vessel mesh for each rod node and segment.
2. **Barrier Function ($B(d)$):** Implement the $C^2$ smooth log-barrier function.
   * If distance $d > \hat{d}$ (activation threshold), $B(d) = 0$.
   * If $d \le \hat{d}$, $B(d) = -(d - \hat{d})^2 \ln(d / \hat{d})$.
3. **Friction (Optional but recommended):** Implement IPC's smooth friction model based on the relative tangential velocity to capture guidewire "stick-slip" behavior.

### Phase 3: Kinematic Actuation (Boundary Conditions)
To navigate the wire, you will apply Dirichlet boundary conditions to the proximal end (node 0).
* **Translation:** Enforce $\Delta x_0$ along the insertion axis.
* **Rotation:** Enforce $\Delta 	heta_0$ (torque) by updating $q_0$.
* Remove these DoFs from the active optimization variables.

### Phase 4: Optimization Solver (Newton's Method)
At each timestep $t$, find the state $x_{t+1}, q_{t+1}$ that minimizes:
$$E_{total} = rac{1}{2h^2} M (x - x_{pred})^2 + E_{elastic}(x, q) + B(d(x))$$
1. **Gradient & Hessian Assembly:** Write Warp kernels to compute the gradient vector and the sparse Hessian matrix.
2. **Linear Solve:** Since Warp does not have a built-in sparse Cholesky solver out-of-the-box for large custom matrices, you have two choices:
   * *Option A (Exact Newton):* Export the sparse matrix to a CPU solver (like SciPy/CHOLMOD) or use a GPU-based sparse solver (e.g., cuSOLVER).
   * *Option B (L-BFGS or Gradient Descent):* Use a first-order optimization method natively in Warp with line-search. L-BFGS is highly effective for C-IPC without requiring full Hessian assembly.
3. **Continuous Collision Detection (CCD):** Implement a filter in the line-search phase to compute the maximum step size $lpha \in [0, 1]$ that guarantees no collisions occur during the update step.

---

## 4. C-IPC vs. XPBD: Benchmarking Strategy

When merging this into your repository, set up the following comparison metrics:

| Metric | Testing Methodology | Expected Outcome |
| :--- | :--- | :--- |
| **Tunneling / Penetration** | Push the guidewire into a sharp bend of the vessel mesh with high force. | **XPBD:** May clip through the wall.<br>**C-IPC:** Guaranteed zero clipping. |
| **Step Time (Performance)** | Measure milliseconds per timestep in Warp. | **XPBD:** Extremely fast (microseconds).<br>**C-IPC:** Slower (milliseconds) due to line-search/CCD. |
| **Energy Dissipation** | Measure total system energy during high-curvature navigation. | **XPBD:** Can artificially gain or lose energy due to constraint projection.<br>**C-IPC:** Physically accurate energy conservation. |
| **Friction / Stick-Slip** | Apply torsion at the proximal end while the distal tip is wedged. | **XPBD:** Struggles with accurate Coulomb friction.<br>**C-IPC:** Highly realistic torsional wind-up and sudden release. |

---

## 5. Development Milestones

1. **Week 1: Cosserat Kinematics in Warp.** Validate that the rod bends and twists correctly in free space (no collisions).
2. **Week 2: Mesh Coupling & Broadphase.** Load the 15k mesh and implement Warp mesh queries to find active contacts.
3. **Week 3: IPC Barrier & Line Search.** Implement the barrier energy and the CCD line-search to prevent tunneling.
4. **Week 4: Actuation & Benchmarking.** Add translation/rotation controls and run side-by-side tests with the XPBD branch.
