import h5py
from flask import Flask, render_template, request
from processFunctions import process_hdf5_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process')
def process():
    process_hdf5_file()
    return render_template('index.html')

if __name__ == '__main__':
    app.run()


