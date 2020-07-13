import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

######### pickle

def save_pickle(filepath, x):
    with open(filepath, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def plot_jasons_scatterplot(x,
                            y, 
                            output_png_path,
                            x_label,
                            y_label,
                            title):
    
    _, ax = plt.subplots()	
    plt.scatter(x,
                y,
                s=3.5,
                )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.savefig(output_png_path, dpi=400)
    plt.clf()