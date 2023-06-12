import h5py
from flask import Flask, render_template, request
from processFunctions import process_hdf5_file, plot_hdf5_image
import plotly.express as px
import plotly.io as pio
import json
import plotly

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process')
def process():
    fig, config = plot_hdf5_image()
    graphJSON = json.dumps({'fig': fig, 'config': config}, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('annotate.html', graphJSON=graphJSON)
   

if __name__ == '__main__':
    app.run()


