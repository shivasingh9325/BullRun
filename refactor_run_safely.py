import os
import shutil
import sys

ROOT = "e:\\Bull_Run"
BACKEND = os.path.join(ROOT, "backend")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

print("Starting refactor...", flush=True)

# 1. ROOT LEVEL MOVES
print("Moving frontend...", flush=True)
# Instead of deleting frontend, just rename the Grow_code/frontend to frontend, avoiding full copy
# First, rename existing frontend to frontend_old to avoid collisions
if os.path.exists(os.path.join(ROOT, "frontend")) and os.path.exists(os.path.join(ROOT, "Grow_code", "frontend")):
    try:
        os.rename(os.path.join(ROOT, "frontend"), os.path.join(ROOT, "frontend_old"))
        print("Renamed existing frontend to frontend_old", flush=True)
    except Exception as e:
        print("Could not rename existing frontend:", e, flush=True)

if os.path.exists(os.path.join(ROOT, "Grow_code", "frontend")):
    try:
        os.rename(os.path.join(ROOT, "Grow_code", "frontend"), os.path.join(ROOT, "frontend"))
        print("Moved Grow_code/frontend to frontend", flush=True)
    except Exception as e:
        print("Could not move Grow_code/frontend:", e, flush=True)

print("Moving config/scripts/data...", flush=True)
for item in ["scripts", "data", "logs", "requirements.txt", ".env"]:
    src = os.path.join(BACKEND, item)
    dst = os.path.join(ROOT, item)
    if os.path.exists(src) and not os.path.exists(dst):
        try:
            os.rename(src, dst)
        except Exception as e:
            print(f"Failed to move {item}:", e, flush=True)

ensure_dir(os.path.join(ROOT, "tests"))
ensure_dir(os.path.join(ROOT, "notebooks"))
ensure_dir(os.path.join(ROOT, "configs"))
if os.path.exists(os.path.join(BACKEND, "configs")):
    try:
        os.rename(os.path.join(BACKEND, "configs"), os.path.join(ROOT, "configs"))
    except:
        pass

# 2. BACKEND RESTRUCTURING
print("Restructuring backend...", flush=True)
NEW_BACKEND = os.path.join(ROOT, "backend_new")
ensure_dir(NEW_BACKEND)

folders = ["api", "core", "models", "pipeline", "services", "backtesting", "broker", "db", "utils"]
for f in folders:
    ensure_dir(os.path.join(NEW_BACKEND, f))

def move_files(src_dir, dst_dir):
    if not os.path.exists(src_dir): return
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(NEW_BACKEND, dst_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
        elif os.path.isdir(s) and item != "__pycache__":
            ensure_dir(d)
            move_files(s, os.path.join(dst_dir, item))

# app/ API mappings
move_files(os.path.join(BACKEND, "app", "api", "v1"), "api")
move_files(os.path.join(BACKEND, "app", "db"), "db")
move_files(os.path.join(BACKEND, "app", "models"), "api") # schemas and schemas.py -> api/
move_files(os.path.join(BACKEND, "app", "services"), "services")
move_files(os.path.join(BACKEND, "app", "core"), "utils") # logging -> utils/
if os.path.exists(os.path.join(BACKEND, "app", "main.py")):
    shutil.copy2(os.path.join(BACKEND, "app", "main.py"), os.path.join(NEW_BACKEND, "api", "main.py"))

# bullrun/ mappings
move_files(os.path.join(BACKEND, "bullrun", "core"), "core")
if os.path.exists(os.path.join(NEW_BACKEND, "core", "pipeline.py")):
    shutil.move(os.path.join(NEW_BACKEND, "core", "pipeline.py"), os.path.join(NEW_BACKEND, "pipeline", "pipeline.py"))

move_files(os.path.join(BACKEND, "bullrun", "models"), "models")
move_files(os.path.join(BACKEND, "bullrun", "backtesting"), "backtesting")
move_files(os.path.join(BACKEND, "bullrun", "utils"), "utils")
move_files(os.path.join(BACKEND, "bullrun", "logic"), "broker")

try:
    os.rename(BACKEND, os.path.join(ROOT, "backend_old"))
    os.rename(NEW_BACKEND, BACKEND)
except Exception as e:
    print("Failed to replace backend dir:", e)

# Create __init__.py files
for root_dir, dirs, files in os.walk(BACKEND):
    init_file = os.path.join(root_dir, "__init__.py")
    if not os.path.exists(init_file):
        open(init_file, 'w').close()

# 3. IMPORT REPLACEMENT
print("Rewriting imports...", flush=True)
def rewrite_imports_in_file(fp):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return
    
    modified = False
    
    replaces = [
        ("from app.api.v1", "from backend.api"),
        ("from app.core", "from backend.utils"),
        ("from app.db", "from backend.db"),
        ("from app.models", "from backend.api"),
        ("import app.models", "import backend.api"),
        ("bullrun.core.pipeline", "backend.pipeline.pipeline"),
        ("bullrun.core.fetcher", "backend.core.fetcher"),
        ("bullrun.core.technical", "backend.core.technical"),
        ("bullrun.models", "backend.models"),
        ("bullrun.logic", "backend.broker"),
        ("bullrun.backtesting", "backend.backtesting"),
        ("bullrun.utils", "backend.utils"),
        ("bullrun.infra", "backend.utils"),
        ("app/api/v1/portfolio", "api/portfolio"),
        ("app.main:app", "backend.api.main:app")
    ]
    
    for old, new in replaces:
        if old in content:
            content = content.replace(old, new)
            modified = True
            
    if modified:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)

for root_dir, _, files in os.walk(ROOT):
    if "node_modules" in root_dir or ".git" in root_dir or "backend_old" in root_dir or "frontend_old" in root_dir or "Grow_code" in root_dir: 
        continue
    for f in files:
        if f.endswith(".py"):
            rewrite_imports_in_file(os.path.join(root_dir, f))

print("Restructuring Complete.", flush=True)
