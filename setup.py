import setuptools
import os

# Function to read the contents of a file (e.g., README.md, requirements.txt)
def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# Read the contents of your README file for long_description
long_description = read_file("README.md")

# Read requirements from requirements.txt
def get_requirements(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Please ensure it exists for dependencies.")
        return []
    return [
        line.strip()
        for line in read_file(filepath).splitlines()
        if line.strip() and not line.startswith("#") # Ignore empty lines and comments
    ]

install_requires = get_requirements("requirements.txt")

setuptools.setup(
    name="ciso-genai", # This is the name your package will be known by on PyPI
    version="0.0.2", # <--- IMPORTANT: Version incremented to reflect changes and ensure unique upload.
                     #      Always use a new version for each upload attempt to TestPyPI/PyPI.
    author="Harsh Bopaliya", # Your Name
    author_email="bopaliyaharsh7@gmail.com", # <--- IMPORTANT: Replace with your actual email
    description="A framework for Causal Intelligence in Multi-Agent Generative AI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshbopaliya/CISO-GENAI-Framework", # Corrected URL format (no Markdown)
    # --- ADDED LICENSE FIELD ---
    license="MIT License", # <--- This line was added/corrected to ensure license metadata is included.
    # --- END ADDED LICENSE FIELD ---
    package_dir={'ciso_genai': 'src'},
    packages=['ciso_genai', 'ciso_genai.envs'],
    # We don't need `setuptools.find_packages` with `where` or `exclude` here,
    # as `package_dir` and explicit `packages` list handle the mapping.
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License", # This classifier is good for filtering on PyPI
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 2 - Pre-Alpha",
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    include_package_data=True, # This tells setuptools to include non-Python files specified in MANIFEST.in
)
