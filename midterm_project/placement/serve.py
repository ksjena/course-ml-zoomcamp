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
response = requests.post(url, json=student) ## post the customer information in json format
result = response.json() ## get the server response
print(result)