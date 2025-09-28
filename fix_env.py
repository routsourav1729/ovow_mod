# fix_environment.py
import yaml

# Read the original environment.yml
with open('environment.yml', 'r') as f:
    env = yaml.safe_load(f)

# Find and fix the pip dependencies
for i, dep in enumerate(env['dependencies']):
    if isinstance(dep, dict) and 'pip' in dep:
        pip_deps = dep['pip']
        # Remove problematic clip package
        pip_deps = [d for d in pip_deps if not d.startswith('clip==')]
        # Add correct CLIP installation
        pip_deps.append('git+https://github.com/openai/CLIP.git')
        dep['pip'] = pip_deps
        break

# Save fixed environment
with open('environment_fixed.yml', 'w') as f:
    yaml.dump(env, f)

print("Fixed environment saved to environment_fixed.yml")