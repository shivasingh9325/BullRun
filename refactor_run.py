import os
import shutil
import glob
import re

ROOT = "e:\\Bull_Run"
BACKEND = os.path.join(ROOT, "backend")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 1. ROOT LEVEL MOVES
print("Moving root level items...")
shutil.rmtree(os.path.join(ROOT, "frontend"), ignore_errors=True)
if os.path.exists(os.path.join(ROOT, "Grow_code", "frontend")):
    shutil.move(os.path.join(ROOT, "Grow_code", "frontend"), os.path.join(ROOT, "frontend"))
shutil.rmtree(os.path.join(ROOT, "Grow_code"), ignore_errors=True)

for item in ["scripts", "data", "logs", "requirements.txt", ".env"]:
    src = os.path.join(BACKEND, item)
    dst = os.path.join(ROOT, item)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    elif os.path.exists(src) and os.path.isdir(src):
        # merge if exists
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        shutil.rmtree(src)

ensure_dir(os.path.join(ROOT, "tests"))
ensure_dir(os.path.join(ROOT, "notebooks"))

# We also have backend/configs. Assuming configs stay in ROOT/configs or inside backend? The layout says no configs at root. Wait, data/ and scripts/ are at root. Let's move configs to ROOT/configs so everything is accessible. Wait, prompt doesn't list configs at all! Let's put configs in backend/configs or ROOT/configs. Actually I'll put configs in ROOT/configs and leave it, we can rename inside code. Wait, the prompt says "Refactor project into...". Let's put configs in backend/core/configs or backend/utils? No, I'll just map them. I will move backend/configs to ROOT/configs.
ensure_dir(os.path.join(ROOT, "configs"))
if os.path.exists(os.path.join(BACKEND, "configs")):
    for f in os.listdir(os.path.join(BACKEND, "configs")):
        shutil.move(os.path.join(BACKEND, "configs", f), os.path.join(ROOT, "configs", f))
    shutil.rmtree(os.path.join(BACKEND, "configs"))

# 2. BACKEND RESTRUCTURING
print("Restructuring backend...")
NEW_BACKEND = os.path.join(ROOT, "backend_new")
ensure_dir(NEW_BACKEND)

folders = ["api", "core", "models", "pipeline", "services", "backtesting", "broker", "db", "utils"]
for f in folders:
    ensure_dir(os.path.join(NEW_BACKEND, f))

# Map old to new
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
move_files(os.path.join(BACKEND, "app", "core"), "utils") # assuming logging is here
if os.path.exists(os.path.join(BACKEND, "app", "main.py")):
    shutil.copy2(os.path.join(BACKEND, "app", "main.py"), os.path.join(NEW_BACKEND, "api", "main.py"))

# bullrun/ mappings
move_files(os.path.join(BACKEND, "bullrun", "core"), "core") # fetcher.py, technical.py. wait, pipeline.py should go to pipeline/
if os.path.exists(os.path.join(NEW_BACKEND, "core", "pipeline.py")):
    shutil.move(os.path.join(NEW_BACKEND, "core", "pipeline.py"), os.path.join(NEW_BACKEND, "pipeline", "pipeline.py"))

move_files(os.path.join(BACKEND, "bullrun", "models"), "models")
move_files(os.path.join(BACKEND, "bullrun", "backtesting"), "backtesting")
move_files(os.path.join(BACKEND, "bullrun", "utils"), "utils")

# logic -> broker (broker_mock, environment)
move_files(os.path.join(BACKEND, "bullrun", "logic"), "broker")

# Replace backend with backend_new
shutil.rmtree(BACKEND)
os.rename(NEW_BACKEND, BACKEND)

# Create __init__.py files
for root_dir, dirs, files in os.walk(BACKEND):
    init_file = os.path.join(root_dir, "__init__.py")
    if not os.path.exists(init_file):
        open(init_file, 'w').close()

# 3. IMPORT REPLACEMENT
print("Rewriting imports...")
def rewrite_imports_in_file(fp):
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    
    # app.* -> ...
    content = content.replace("from backend.api.portfolio", "from backend.api.portfolio")
    content = content.replace("from backend.utils.logging", "from backend.utils.logging")
    content = content.replace("from backend.db", "from backend.db")
    content = content.replace("from backend.api.schemas", "from backend.api.schemas")
    content = content.replace("import backend.api.schemas", "import backend.api.schemas")
    
    # bullrun.* -> ...
    content = content.replace("backend.pipeline.pipeline", "backend.pipeline.pipeline")
    content = content.replace("backend.core.fetcher", "backend.core.fetcher")
    content = content.replace("backend.core.technical", "backend.core.technical")
    
    content = content.replace("backend.models.technical_model", "backend.models.technical_model")
    content = content.replace("backend.models.meta", "backend.models.meta")
    content = content.replace("backend.models.sentiment_model", "backend.models.sentiment_model")
    content = content.replace("backend.models.agent", "backend.models.agent")
    content = content.replace("backend.models.base", "backend.models.base")
    
    content = content.replace("backend.broker.broker_mock", "backend.broker.broker_mock")
    content = content.replace("backend.broker.environment", "backend.broker.environment")
    
    content = content.replace("backend.backtesting.engine", "backend.backtesting.engine")
    
    content = content.replace("backend.utils.logger", "backend.utils.logger")
    content = content.replace("backend.utils.diagnostic", "backend.utils.diagnostic")
    
    # Some standalone path string replacements
    content = content.replace("api/portfolio", "api/portfolio")
    content = content.replace("backend.api.main:app", "backend.api.main:app")
    
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)

# Scan all python files in ROOT
for root_dir, _, files in os.walk(ROOT):
    if "node_modules" in root_dir or ".git" in root_dir: continue
    for f in files:
        if f.endswith(".py"):
            rewrite_imports_in_file(os.path.join(root_dir, f))

print("Restructuring Complete.")
