echo [$(date)]: "START"


echo [$(date)]: "creating env file with python 3.8 version"


conda create -p ./llm_envn python=3.8 -y


source avtivate ./llm_envn


echo [$(date)]: "installing the dev requirements"


pip install -r requirements_dev.txt


echo [$(date)]: "END"