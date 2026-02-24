# #1 : 3D Modeling in CS class
\>>> By. **Eun-Sung Choi**(David C.)  

Started: `Feb 1, 2026` → Ended: `Feb 23, 2026`  

> Interactive 3D scene editor built with [CMU Graphics](https://academy.cs.cmu.edu/docs).  
> Create and manipulate Cuboid, Cube, Sphere in world space with keyboard controls.  
> Camera movement and depth-based rendering included.

---

## Requirements & Run

- **Python** 3.11 or 3.12
- **Dependencies:** `pip install -r requirements.txt`
- **Run:** `python mat3.py`

---

## TO DO
+ [x] Multi Object Friendly Design
+ [x] Graphics Group Management
+ [x] Keyboard Binding & Interface
+ [x] Rotation
+ [x] Movable Cam
+ [x] Depth & Layer
+ [ ] ~~Intuitive Cam Move~~
+ [x] Additional Shapes
    + [x] Sphere
    + [ ] *(Optional)* *Something else...*
+ [x] Add More Arguments
+ [ ] *(Optional)* Make It as a **Library**
+ [ ] *(Optional)* Try To Make Something Using **Quaternion**
---

## Built-in API

### Constants / Globals
| Name | Type | Description |
|------|------|-------------|
| `CMU_RUN` | bool | If `True`, runs `c.run()` automatically |
| `f` | float | Focus (default 250) |
| `MIN_SIZE` | float | Minimum size (1e-6) |
| `NEAR_PLANE` | float | Near plane (1.0) |
| `screen_center` | tuple[float, float] | Screen center (200, 200) |
| `cam_cord` | Vec3 | Camera position (global) |
| `cam_rotation` | Vec3 | Camera rotation (global) |
| `camera` | Camera | Camera instance |
| `depth_layer` | DepthLayerManager | Render order manager |
| `selected_object_info` | SelectedObjectInfo | Selected object UI |
| `input_info` | InputInfo | Input state UI |
| `camera_info` | CameraInfo | Camera info UI |
| `SHOW_AXIS` | bool | Axis visibility toggle |

---

### Data Types (dataclass)
| Class | Fields | Description |
|-------|--------|-------------|
| **`Vec3`** | `x, y, z: float` | 3D vector. Supports `+`, `-` |
| **`Vec2`** | `x, y: float` | 2D vector. `+`, `-`, `dot()`, `norm()` |
| **`Mat3`** | `m: tuple` | 3×3 matrix. `@` (Mat3@Vec3, Mat3@Mat3) |
| **`Mat2`** | `m: tuple` | 2×2 matrix. `@`, `eigen()` |

---

### Utility Functions
| Function | Signature | Description |
|----------|-----------|-------------|
| `v2s` | `(v: Vec3) -> tuple[float, float, float]` | Vec3 → (x, y, z) |
| `s2v` | `(s: float) -> Vec3` | Scalar → Vec3(s, s, s) |

---

### Rotation / Projection
| Function | Signature | Description |
|----------|-----------|-------------|
| `Rx` | `(theta: float) -> Mat3` | X-axis rotation matrix (rad) |
| `Ry` | `(theta: float) -> Mat3` | Y-axis rotation matrix (rad) |
| `Rz` | `(theta: float) -> Mat3` | Z-axis rotation matrix (rad) |
| `rotateVerts` | `(verts: list[Vec3], rotation: Vec3) -> list[Vec3]` | Rotate vertex list |
| `projectWorldToCamera` | `(points: Vec3, cam_cord?, cam_rotation?) -> Vec3` | World → camera coords |
| `projectCameraToScreen` | `(points: Vec3, screen_center?) -> tuple[float, float] \| None` | Camera → screen coords |

---

### 3D Primitives (returns CMU Graphics shapes)
| Function | Signature | Description |
|----------|-----------|-------------|
| `Line` | `(p0, p1: Vec3, cam_cord?, cam_rotation?, screen_center?, **kwargs) -> c.Line \| None` | 3D segment → 2D Line |
| `Polygon` | `(verts: list[Vec3], cam_cord?, cam_rotation?, screen_center?, **kwargs) -> c.Polygon \| None` | 3D vertices → 2D Polygon |
| `Oval` | `(cord, s, t: Vec3, cam_cord?, cam_rotation?, screen_center?, **kwargs) -> c.Oval \| None` | 3D ellipse → 2D Oval |

---

### Object Management
| Function | Signature | Description |
|----------|-----------|-------------|
| `register_object` | `(obj, name: str, obj_type: str \| None = None)` | Register object to scene |
| `get_selected_object` | `(kind: str) -> object \| str \| None` | `kind`: `'obj'` \| `'name'` \| `'type'` |
| `select_next_object` | `()` | Select next object |
| `clearObject` | `(name: str)` | Remove object from scene |

---

### Axis / Visualization
| Function | Signature | Description |
|----------|-----------|-------------|
| `drawAxis` | `() -> c.Group \| None` | Draw coordinate axes (respects SHOW_AXIS) |

---

### 3D Shape Classes

#### Cuboid
- **Constructor:** `Cuboid(cord, size: Vec3, rotation=None, cam_cord?, cam_rotation?, screen_center?, **kwargs)`
- **Methods:** `redraw()`, `move(delta)`, `set_cord(cord)`, `reset_cord()`, `scale(delta)`, `set_size(size)`, `reset_size()`, `rotate(delta)`, `set_rotation(rotation)`, `reset_rotation()`
- **Attributes:** `cord`, `size`, `rotation`, `group`, `kwargs`
- **kwargs:** `fill`, `border`, `borderWidth`, `opacity`, `visible` (Group: `visible`, `opacity`)

#### Cube
- **Constructor:** `Cube(cord: Vec3, size: float, rotation=None, ...)`
- **Methods:** Same as Cuboid (scale is uniform)
- **Attributes:** Same as Cuboid

#### Sphere
- **Constructor:** `Sphere(cord: Vec3, size: float, rotation=None, ...)`
- **Methods:** Same as Cuboid
- **Attributes:** Same as Cuboid

---

### Camera
- **Constructor:** `Camera(cord: Vec3, rotation: Vec3)`
- **Methods:** `move(delta)`, `set_cord(cord)`, `reset_cord()`, `rotate(delta)`, `set_rotation(rotation)`, `reset_rotation()`
- **Attributes:** `cord`, `rotation`

---

### DepthLayerManager
- **Constructor:** `DepthLayerManager()`
- **Methods:** `update(obj_registry, cam_cord, cam_rotation, axis_group?, top_groups?)` — Update render order by camera distance

---

### UI Functions
| Function | Signature | Description |
|----------|-----------|-------------|
| `ask` | `(prompt: str) -> str` | Text input dialog |
| `alert` | `(message: str \| None)` | Message popup |

---

### UI Classes
| Class | Constructor | Methods | Description |
|-------|-------------|---------|--------------|
| **InputInfo** | `(key_x?, key_y?, mouse_x?, mouse_y?)` | `set_keyboard(keys?, modifiers?)`, `set_mouse(x?, y?, button?)`, `set_mouse_pressed(pressed)` | Keyboard/mouse state display |
| **CameraInfo** | `(x?, y?)` | `update()` | Camera position/rotation display |
| **SelectedObjectInfo** | `(x?, y?)` | `update()` | Selected object info display |

---

### Internal Functions (Private, for extension)
| Function | Description |
|----------|-------------|
| `_clamp_size`, `_clamp_vec3_size` | Enforce minimum size |
| `_split_kwargs`, `_apply_group_kwargs` | Split kwargs into Group/child |
| `_cuboid_faces` | Generate Cuboid faces |
| `_sphere_silhouette`, `_sphere_ovals` | Sphere silhouette/oval generation |
| `_rotateWorldToCamera` | Direction vector world→camera transform |
| `_clip_segment_to_near_plane` | Segment near-plane clipping |
| `_oval_params_from_camera` | Camera-space oval → screen params |

---



## KEY BINDINGS
> *All **transformations**(translation, scaling, rotation, etc.) are based on the **WORLD AXIS.***
- <kbd>TAB</kbd> : **SELECT** objects
- <kbd>BACKSPACE</kbd> : **DELETE** selected object
- <kbd>SHIFT</kbd> + <kbd>=</kbd> : **ADD** object (*Default `Cuboid(s2v(0), s2v(50))`*)
- <kbd>g</kbd> : Toggle **AXIS VISIBILITY**
- Reset
    - **OBJECT**
        - <kbd>t</kbd> : Reset object **POSITION**
        - <kbd>SHIFT</kbd> + <kbd>t</kbd> : Reset object **ROTATION**
    - **CAMERA**
        - <kbd>b</kbd> : Reset camera **POSITION**
        - <kbd>SHIFT</kbd> + <kbd>b</kbd> : Reset camera **ROTATION**
- **OBJECT**
    - Translation
        - <kbd>a</kbd> : Move **LEFT** (X-axis)
        - <kbd>d</kbd> : Move **RIGHT** (X-axis)
        - <kbd>w</kbd> : Move **UP** (Y-axis)
        - <kbd>s</kbd> : Move **DOWN** (Y-axis)
        - <kbd>z</kbd> : Move **FRONT** (Z-axis)
        - <kbd>x</kbd> : Move **BACK** (Z-axis)
    - Scale
        - <kbd>SHIFT</kbd> + <kbd>a</kbd> : Scale **WIDTH +** (X-axis)
        - <kbd>SHIFT</kbd> + <kbd>d</kbd> : Scale **WIDTH -** (X-axis)
        - <kbd>SHIFT</kbd> + <kbd>w</kbd> : Scale **HEIGHT +** (Y-axis)
        - <kbd>SHIFT</kbd> + <kbd>s</kbd> : Scale **HEIGHT -** (Y-axis)
        - <kbd>SHIFT</kbd> + <kbd>z</kbd> : Scale **DEPTH +** (Z-axis)
        - <kbd>SHIFT</kbd> + <kbd>x</kbd> : Scale **DEPTH -** (Z-axis)
    - Rotate
        - <kbd>r</kbd> : Rotate **PITCH +** (X-axis)
        - <kbd>f</kbd> : Rotate **PITCH -** (X-axis)
        - <kbd>c</kbd> : Rotate **YAW +** (Y-axis)
        - <kbd>v</kbd> : Rotate **YAW -** (Y-axis)
        - <kbd>q</kbd> : Rotate **ROLL +** (Z-axis)
        - <kbd>e</kbd> : Rotate **ROLL -** (Z-axis)

- **CAMERA**
    - Translation
        - <kbd>j</kbd> : Move **LEFT** (X-axis)
        - <kbd>l</kbd> : Move **RIGHT** (X-axis)
        - <kbd>i</kbd> : Move **UP** (Y-axis)
        - <kbd>k</kbd> : Move **DOWN** (Y-axis)
        - <kbd>.</kbd> : Move **FRONT** (Z-axis)
        - <kbd>,</kbd> : Move **BACK** (Z-axis)
    - Rotate
        - <kbd>h</kbd> : Rotate **PITCH +** (X-axis)
        - <kbd>y</kbd> : Rotate **PITCH -** (X-axis)
        - <kbd>m</kbd> : Rotate **YAW +** (Y-axis)
        - <kbd>n</kbd> : Rotate **YAW -** (Y-axis)
        - <kbd>u</kbd> : Rotate **ROLL +** (Z-axis)
        - <kbd>o</kbd> : Rotate **ROLL -** (Z-axis)


---

## About CMU Graphics Library

*Find more info from [CMU Graphics Docs](https://academy.cs.cmu.edu/docs)*

This project uses `cmu_graphics` for 2D rendering. 3D objects are projected to screen via `mat3.py`. To add your own shapes in `mat3.py`:

```python
from cmu_graphics import cmu_graphics as c
# ... (import Vec3, s2v, Cuboid, Cube, Sphere, register_object from mat3)

cuboid = Cuboid(s2v(0), s2v(50), fill='salmon', opacity=70)
register_object(cuboid, 'my_cuboid')
# c.run() is called at the end of mat3.py
```
