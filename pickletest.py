import pickle
from policy import StateActionProgram


a = StateActionProgram("a","hello")
with open("test.pkl", 'wb') as f:
    pickle.dump(a, f)

with open("test.pkl", 'rb') as f:
    a = pickle.load(f)

print(vars(a))

