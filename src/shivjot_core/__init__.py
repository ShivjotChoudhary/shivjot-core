import pandas as pd
import numpy as np

# This part adds the ".shivjot" tool to Pandas
@pd.api.extensions.register_dataframe_accessor("shivjot")
class ShivjotData:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def analyze(self):
        """Your custom AI data analysis tool."""
        print("--- Shivjot-Core AI Analysis ---")
        return self._obj.describe()

# This is your Image Processing tool
def convert_to_frequency(image_matrix):
    """Converts an image to the frequency domain."""
    return np.fft.fft2(image_matrix)