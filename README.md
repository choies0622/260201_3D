# #1 : 3D Modeling in CS class
\>>> By. **Eun-Sung Choi**(David C.)  

Started: `Feb 1, 2026`  
Ended: `Feb 23, 2026`  

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
+ [ ] *(Optional)* Try To Make Samething Using **Quaternion**
---

# Docs
## KEY BINDINGS
> *All **transformations**(translation, scaling, rotation, etc.) are based on the **WORLD AXIS.***
- <kbd>TAB</kbd> : **SELECT** objects
- <kbd>BACKSPACE</kbd> : **DELETE** selected object
- <kbd>SHIFT</kbd> + <kbd>=</kbd> : **ADD** object (*Default `Cuboid(0, 0, 0, 50, 50, 50)`*)
- <kbd>g</kbd> : Toggle **AXIS VISIBLITY**
- Reset
    - **OBJECT**
        - <kbd>t</kbd> : Reset object **POSITION**
        - <kbd>SHIFT</kbd> + <kbd>t</kbd> : Reset object **ROTATION**
    - **CAMARA**
        - <kbd>b</kbd> : Reset camara **POSITION**
        - <kbd>SHIFT</kbd> + <kbd>b</kbd> : Reset camara **ROTATION**
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

- **CAMARA**
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

## About CMU Graphics Library
*Find more info from [CMU Graphics Docs](https://academy.cs.cmu.edu/docs)*

### Import Library
```python
# from cmu_graphics import *
from cmu_graphics import cmu_graphics as c

#### YOUR CODES WILL BE PLACED HERE ####

c.run()
```
