from setuptools import find_packages,setup
from typing import List


with open("README.md", "r", encoding="utf-8") as f:
    long_descriptions = f.read()

AUTHOR_USER_NAME = "Tran Phi Hung"
__version__ = "0.0.1"
NAME_PROJECT = "VisualQuestionAnswering"
description = "A project using both image and text is sovled some problem related to nlp tasks. "
REPO_NAME = "tph-kds"
AUTHOR_EMAIL = "tranphihung8383@gmail.com"

setup(
    name= NAME_PROJECT,
    version= __version__ ,
    author= AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description= description,
    long_description= long_descriptions,
    long_description_content_type="text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls= {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues", 
    },
    package_dir={"": "src"},
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages(where="src"),
)