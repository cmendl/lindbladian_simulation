from setuptools import setup


setup(
    name="lindbladian_sim",
    version="0.0.1",
    author="Christian B. Mendl",
    author_email="christian.b.mendl@gmail.com",
    packages=["lindbladian_sim"],
    url="https://github.com/cmendl/lindbladian_simulation",
    install_requires=[
        "numpy",
        "scipy",
    ],
)
