from setuptools import setup, find_packages

setup(
    name="stat_ait",
    version="0.1.0",
    description="An attribute inference test using statistical inference",
    author="Xenia F. Gerloff",
    license="MIT",
    packages=find_packages(
        include=["stat_ait", "stat_ait.*", "standard_tests", "standard_tests.*"]
    ),
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "run=scripts.run:main",
            "run-standard-tests=scripts.run_standard_tests:main",
        ],
    },
    python_requires=">=3.10",
)
