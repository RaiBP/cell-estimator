import h5py
from flask import Flask, render_template, request, jsonify
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

@app.route('/annotations-data', methods=['POST'])
def handle_annotations_data():
    # Get the annotation data from the request
    relayout_data = request.json

    # Check if the annotation data contains shapes
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]

        # Check if shapes exist
        if shapes:
            latest_shape = shapes[-1]
            shape_type = latest_shape.get("type")
            print(shape_type)  # Print the shape type of the latest annotation in the terminal
            if shape_type == "path":
                shape_path = latest_shape.get("path")
                print(f'shape: {shape_path}')
            elif shape_type == "rect":
                rect_x0 = latest_shape.get("x0")
                rect_y0 = latest_shape.get("y0")
                rect_x1 = latest_shape.get("x1")
                rect_y1 = latest_shape.get("y1")
                print(f'x0: {rect_x0}')
                print(f'y0: {rect_y0}')
                print(f'x1: {rect_x1}')
                print(f'y1: {rect_y1}')
            else:
                print("")
        else:
            print("Mode bar clicked")  # Print a message if no shapes are found
    else:
        print("No shapes found")  # Print a message if no shapes are found

    return jsonify({'message': 'Annotations processed successfully'})

if __name__ == '__main__':
    app.run()


