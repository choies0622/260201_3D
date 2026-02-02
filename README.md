# #1 : 3D Modeling in CS class
\>>> By. **Eun-Sung Choi**, David C.  
Proj. Started On `Feb 2, 2026`.  

---
*something will be updated below here*

## Setup (macOS)

### Recommended (Homebrew + venv)

This avoids mixing multiple system Pythons and keeps dependencies per-project.

```bash
brew install python@3.13
```

```bash
cd /Users/choies/Documents/_CODESPACE/260201_3d
/usr/local/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python untitled.py
```

### Tkinter troubleshooting (Homebrew Python)

If you see:

- `ModuleNotFoundError: No module named '_tkinter'`

your current Python build does not include Tk support. This is **not** fixable via `pip`.

#### Option A (recommended for Homebrew users): build a Tk-enabled Python via `pyenv` (installed by brew)

```bash
brew install tcl-tk pyenv
```

Make sure your shell initializes `pyenv` (put this in `~/.zshrc`, then restart the shell):

```bash
eval "$(pyenv init -)"
```

Build Python with Homebrew `tcl-tk` (the env vars are important), then create a fresh venv:

```bash
export PATH="$(brew --prefix tcl-tk)/bin:$PATH"
export LDFLAGS="-L$(brew --prefix tcl-tk)/lib"
export CPPFLAGS="-I$(brew --prefix tcl-tk)/include"
export PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig"

pyenv install 3.13.11
pyenv local 3.13.11

python -m venv .venv
source .venv/bin/activate
python -c "import tkinter as tk; tk.Tk().destroy(); print('tk OK')"
python untitled.py
```

#### Option B: reinstall Homebrew `python@3.13` (may still not provide Tk in some setups)

Some environments require rebuilding `python@3.13` from source after installing `tcl-tk`:

```bash
brew install tcl-tk
brew reinstall python@3.13 --build-from-source
```

Then recreate the venv and verify:

```bash
rm -rf .venv
/usr/local/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -c "import tkinter as tk; tk.Tk().destroy(); print('tk OK')"
```

### Quick (use existing python3)

If `python` is not found, use **`python3`** / **`python3 -m pip`**:

```bash
python3 --version
python3 -m pip install -r requirements.txt
python3 untitled.py
```

If you want `python` to work, you can also add an alias in your shell config:

```bash
alias python=python3
```
