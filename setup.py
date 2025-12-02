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
    name="ciso-genai",
    version="0.1.0",  # Updated version with working credit assignment
    author="Harsh Bopaliya",
    author_email="bopaliyaharsh7@gmail.com",
    description="Causal Credit Assignment for Multi-Agent AI Systems - Determine which agent caused the reward",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshbopaliya/CISO-GENAI-Framework",
    license="MIT License",
    package_dir={'ciso_genai': 'src'},
    packages=['ciso_genai', 'ciso_genai.envs'],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'torch': ['torch>=2.0.0'],
        'full': ['torch>=2.0.0', 'gymnasium>=0.29.0'],
    },
    include_package_data=True,
    keywords=[
        'multi-agent',
        'reinforcement-learning', 
        'credit-assignment',
        'causal-inference',
        'shapley-values',
        'marl',
        'ai',
    ],
)
