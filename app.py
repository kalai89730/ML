from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)

with open('linear_model.pkl', 'rb') as file:
  model = pickle.load(file)


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  experience = request.form['exp']

  input_data = [[float(experience)]]
  reshaped_data=np.asarray(input_data).reshape(1,-1)
  prediction = model.predict(reshaped_data)

  return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
  app.run(host='0.0.0.0',debug=True)
