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
    version="0.0.1", # <--- IMPORTANT: Increment the version (e.g., to 0.0.8 or 0.1.0)
                     #      Always use a new version for each upload attempt to TestPyPI/PyPI.
    author="Harsh Bopaliya", # Your Name
    author_email="bopaliyaharsh7@gmail.com", # <--- IMPORTANT: Replace with your actual email
    description="A framework for Causal Intelligence in Multi-Agent Generative AI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshbopaliya/CISO-GENAI-Framework", # Corrected URL format (no Markdown)
    # --- CRITICAL CONFIGURATION FOR YOUR CURRENT STRUCTURE ---
    # This tells setuptools: "When you build the 'ciso_genai' package,
    # its content comes from the 'src' directory in my local project."
    package_dir={'ciso_genai': 'src'},
    # This explicitly lists the top-level package and its sub-packages
    # that should be installed *under the 'ciso_genai' namespace*.
    # 'ciso_genai' maps to 'src/'
    # 'ciso_genai.envs' maps to 'src/envs/'
    packages=['ciso_genai', 'ciso_genai.envs'],
    # We don't need `setuptools.find_packages` with `where` or `exclude` here,
    # as `package_dir` and explicit `packages` list handle the mapping.
    # --- END CRITICAL CONFIGURATION ---
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
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
