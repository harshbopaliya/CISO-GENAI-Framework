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
    version="0.0.1", # Start with a development version (major.minor.patch)
    author="Harsh Bopaliya", # Your Name
    author_email="bopaliyaharsh7@gmail.com", # <--- IMPORTANT: Replace with your actual email
    description="A framework for Causal Intelligence in Multi-Agent Generative AI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshbopaliya/CISO-GENAI-Framework", # <--- IMPORTANT: Replace with your actual GitHub repo URL
    packages=setuptools.find_packages(exclude=["agent_demo*", "tests*"]), # Automatically finds Python packages (folders with __init__.py)
    # Changed: Added 'exclude=["agent_demo*", "tests*"]' to ensure the demo folder
    # and any potential test folders are NOT installed as part of the framework.
    # This assumes 'agent_demo' is purely for demonstration and not a module.
    # If agent_demo should be installable, remove "agent_demo*" from exclude.
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12", # Added Python 3.12 classifier
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 2 - Pre-Alpha", # 1 - Planning, 2 - Pre-Alpha, 3 - Alpha, 4 - Beta, 5 - Production/Stable
    ],
    python_requires='>=3.8', # Minimum required Python version
    install_requires=install_requires, # List of dependencies from requirements.txt
    include_package_data=True, # <--- IMPORTANT: Added this line.
    # This tells setuptools to look for a MANIFEST.in file and include
    # the non-Python files specified there (like your .yaml config files)
    # when building the wheel and installing the package.
)

