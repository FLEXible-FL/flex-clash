from setuptools import find_packages, setup

setup(
    name="flex-clash",
    version="0.0.1",
    author="Nuria Rodríguez Barroso",
    keywords="adversarial attacks defences FL federated-learning flexible",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["flex"],
)