# Engineering Placements Prediction
# Predict Final Year Engineering College Placements

A University Announced Its On-Campus Placement Records For The Engineering Course. The Data Is From The Years 2013 And 2014.
The Following Is The College Placements Data Compiled Over 2 years. **Use This Data To Predict And Analyse Whether A Student Gets Placed**, Based On His/Her Background.
Perform Extensive EDAs And Bring Out Insights.
Build classification model using various ML techniques

* [https://www.kaggle.com/tejashvi14/engineering-placements-prediction?select=collegePlace.csv](Kaggle dataset reference)

* Dataset: kaggle datasets download -d tejashvi14/engineering-placements-prediction

# Download Code
* course-ml-zoomcamp --> midterm_project --> placement
* pip install pipenv
* pipenv install

# start the server
* linux: gunicorn --bind 0.0.0.0:9696 predict:app
* windows: waitress-serve --listen=0.0.0.0:9696 predict:app

# Predict sample dict
Code: [serve.py](serve.py)

student = {
    "age": 22,
    "gender": "female",
    "stream": "computerscience",
    "internships": 0,
    "cgpa": 8,
    "hostel": 0,
    "historyofbacklogs": 0
}

import requests ## to use the POST method we use a library named requests
url = 'http://localhost:9696/predict' ## this is the route we made for prediction
response = requests.post(url, json=student) ## post the student information in json format
result = response.json() ## get the server response
print(result)

# docker details
* FROM python:3.8.12-slim
* WORKDIR /app
* RUN pip install pipenv
* COPY Pipfile Pipfile.lock ./
* RUN pipenv install --deploy --system
* COPY predict.py model.bin ./
* EXPOSE 9696
* ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]


* dokcer local: docker build -t placement-prediction .

* dokcer tagging: docker tag placement-prediction:latest ksjenar01/ml-in-action:placement-prediction
* docker push to docker hub: docker push ksjenar01/ml-in-action:placement-prediction