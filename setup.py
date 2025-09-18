from __future__ import annotations

from setuptools import find_packages
from setuptools import setup


def read_constraints():
    """Read constraints.txt and return set of blocked packages"""
    blocked_packages = set()
    try:
        with open("constraints.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name from constraint (e.g., "torch==0.0.0" -> "torch")
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    if "==0.0.0" in line:  # Only block packages with version 0.0.0
                        blocked_packages.add(package_name.lower())
    except FileNotFoundError:
        print("'constraints.txt' not found, no packages will be blocked")

    return blocked_packages


# Read requirements.txt, ignore comments and apply constraints
try:
    REQUIRES = list()
    blocked_packages = read_constraints()

    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            # Extract package name (before any version specifiers)
            package_name = \
            line.split('>=')[0].split('==')[0].split('<=')[0].split('>')[0].split('<')[0].split('!=')[0].split('[')[
                0].strip()

            # Check if package is blocked in constraints.txt
            if package_name.lower() not in blocked_packages:
                REQUIRES.append(line)
            else:
                print(f"Blocking package from constraints.txt: {package_name}")

except FileNotFoundError:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name="FinRL",
    version="0.3.8",
    include_package_data=True,
    author="AI4Finance Foundation",
    author_email="contact@ai4finance.org",
    url="https://github.com/AI4Finance-Foundation/FinRL",
    license="MIT",
    packages=find_packages(),
    description="FinRL: Financial Reinforcement Learning Framework.",
    long_description="Version 0.3.5 notes: stable version, code refactoring, more tutorials, clear documentation",
    # It is developed by `AI4Finance`_. \
    # _AI4Finance: https://ai4finance.org/",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcement Learning, Finance",
    platform=["any"],
    python_requires=">=3.7",
    install_requires=REQUIRES,
)