from flask import Flask, request, render_template, url_for
import os
from helper import *


#  configuring device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def load_models():
    #  loading 75px model
    model_75x = CarRecognition75()
    model_75x.load_state_dict(torch.load('model_states/model_state75.pt',
                                         map_location=device))
    #  loading 100px model
    model_100x = CarRecognition100()
    model_100x.load_state_dict(torch.load('model_states/model_state100.pt',
                                          map_location=device))

    #  instantiating ensemble
    model = EnsembleModels(model_75x, model_100x)
    return model


#  loading models
model_ex = load_models()


#  flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('predict.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        image.save('to_pred.jpg', int(2e+6))
        output = model_ex.average_confidence('to_pred.jpg')
        os.remove('to_pred.jpg')
        return render_template('predict.html', output=f'OUTPUT: {output}')
    else:
        return home()


app.run()
