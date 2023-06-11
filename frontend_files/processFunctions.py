import h5py
from tkinter import Tk, filedialog

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
    
