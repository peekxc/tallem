# %% Imports
import numpy as np 
import cppimport.import_hook


#%% Compile and import src files at runtime using cppimport 
carma_svd = cppimport.imp_from_filepath("src/carma_svd.cpp")

# %% Random data 
X = np.random.uniform(size=(10, 2))
