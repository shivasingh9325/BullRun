import os
import re

backend_dir = r"e:\Bull_Run\backend"

replacements = [
    (r'from bullrun\.core', 'from backend.core'),
    (r'from bullrun\.models', 'from backend.models'),
    (r'from bullrun\.logic', 'from backend.broker'),
    (r'from bullrun\.infra', 'from backend.utils'), # guess
    (r'from bullrun\.utils', 'from backend.utils'),
    (r'from bullrun\.backtesting', 'from backend.backtesting'),
    (r'import bullrun\.', 'import backend.'),
    (r'from app\.db', 'from backend.db'),
    (r'from app\.api', 'from backend.api'),
    (r'from app\.core', 'from backend.core'),
    (r'from app\.models', 'from backend.api'), # schemas usually? Let's verify
    (r'app\.main', 'backend.api.main'), # for uvicorn snippet
]

for root, dirs, files in os.walk(backend_dir):
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            orig_content = content
            for pat, repl in replacements:
                content = re.sub(pat, repl, content)
                
            if content != orig_content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"Updated {filepath}")
