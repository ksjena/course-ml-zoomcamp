import pickle

# loading random forest model
with open('midterm_project\placement\model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

student = {
    "age": 30,
    "gender": "male",
    "stream": "mechanical",
    "internships": 0,
    "cgpa": 9,
    "hostel": 1,
    "historyofbacklogs": 0
}

X = dv.transform([student])
y_pred = model.predict_proba(X)[0, 1]

print('input:', student)
print('output:', y_pred)
