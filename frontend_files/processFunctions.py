import h5py
from tkinter import Tk, filedialog
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as offline

def process_hdf5_file():
    # Open a file dialog for the user to select an HDF5 file
    Tk().withdraw()
    filepath = filedialog.askopenfilename(title="Select File")

    # Check if a file was selected
    if not filepath:
        print("No file selected.")
        return

    try:
        # Open the HDF5 file
        with h5py.File(filepath, "r") as file:
            #Print all dataset names
            print("Dataset names:")
            for dataset_name in file.keys():
                print(dataset_name)

        print("File processed successfully.")
    except Exception as e:
        print("An error occurred while processing the file:")
        print(e)


def plot_hdf5_image():
    # Open a file dialog for the user to select an HDF5 file
    Tk().withdraw()
    filepath = filedialog.askopenfilename(title="Select File")

    # Check if a file was selected
    if not filepath:
        print("No file selected.")
        return

    try:
        # Open the HDF5 file and plot a phase image
        with h5py.File(filepath, "r") as file:
            dataset = file['phase/images']
            # Get the image
            img = dataset[10]

            fig = px.imshow(img)
            fig.update_layout(dragmode="drawclosedpath")
            
            # Edit here the configuration of the image
            config = {
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ],
                "displaylogo": False
            }
            
        print("File processed successfully.")
        return fig, config
    except Exception as e:
        print("An error occurred while processing the file:")
        print(e)
    
