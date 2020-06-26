import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rcParams, font_manager
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

insert_matrix = [
                    [0.7,0.3,0.3,0.6,-1,-1.5],
                    [-0.1,0.6,0.5,0.3,0,-0.2],
                    [1,1.2,1.4,1.2,1.6,1.3],
                    [0.9,1.3,1.6,1.6,1.9,1.7],
                    [1.2,1.1,1.6,1.5,1.9,1.9],
                    [1.4,1.3,1.7,1.9,2.2,2.2],
                    [1.4,1.1,1.6,1.9,2.1,2.1],
                    [1.2,0.9,1.2,1.5,1.7,1.9],
                    [1,0.7,1,1.1,1.4,1.7],
                    [0.1,-0.3,0.1,0.6,0.7,0.5],
                ]

swap_matrix = [
                    [1.6,1,0.9,-0.1,-0.8,-1.3],
                    [0.6,0.3,-0.2,-0.2,-0.7,-1.3],
                    [1.9,2,1.8,1.5,1.5,1.2],
                    [1.9,2.2,2.2,1.7,2.1,1.5],
                    [2.3,1.7,2,1.4,2.1,1.6],
                    [2.1,2.3,2.6,2.1,2.7,1.9],
                    [1.9,2,2.3,2,2.3,2],
                    [1.7,1.6,2.1,1.9,1.9,1.7],
                    [1.1,1.3,1.1,1.2,1.4,1.5],
                    [0.3,0.3,0.5,0.7,0.4,0.1],
                ]

both_matrix = [ [2.2,2.1,1.9,1,0.3,0],
                [0.9,1.2,1.1,0.8,-0.1,0.6],
                [2.1,2,2.7,1.7,1.8,1.5],
                [1.3,1.4,2,2.4,1.9,1.9],
                [1.2,1.2,1.6,1.5,1.9,1.9],
                [1.7,1.6,2,1.9,1.3,1.3],
                [1.2,1.4,1.2,1.6,1.2,1.3],
                [1.1,0.8,0.7,1.1,0.7,1.1],
                [0.6,0.2,0.7,0.4,0.5,0.3],
                [-0.2,-0.4,-0.4,-0.3,-0.5,-0.2]
                ]   

#Generates a heatmap from a confusion matrix
def generate_heatmap(confusion_matrix, output_file, cbar_kw={}, cbarlabel="", **kwargs):
    # confusion_matrix = [[round(item*1.0 / np.sum(subl), 2) for item in subl] for subl in confusion_matrix]
    #What you want to label your rows and columns

    confusion_matrix = np.array(confusion_matrix).T
    confusion_matrix = np.flip(confusion_matrix, axis=0)

    row_labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    col_labels = list(reversed(["0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]))
    #Font to use
    csfont = {'fontname': 'Arial'}
    ticks_font = font_manager.FontProperties(family = 'Arial')
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, **kwargs)
    im.set_clim(-2, 3)
    #can specify the color hues to use (google this if you want to change from blues), change in main
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.set_clim(-3.0, 3.0)
    #Puts the color bar to indicate what each shade of color means
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #Manually assign the ticks 
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_yticks(np.arange(len(col_labels)))
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels)
    #Set these as your axes labels
    # ax.set_xlabel('Original Data Weight', fontsize=12, **csfont)
    # ax.set_ylabel('Augmentation Strength', fontsize=12, **csfont)
    text_matrices = np.empty((len(confusion_matrix), len(confusion_matrix[0])), dtype=object)
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            if confusion_matrix[i][j] == 0.0:
                text_matrices[i][j] = '0.0'
            else:
                text_matrices[i][j] = str(confusion_matrix[i][j])#.lstrip('0')
    #Include text labels for each square
    threshold = 1.3
    textcolors = ["black", "white"]
    kw = dict(ha = 'center', va = 'center', fontsize = 12)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            kw.update(color = textcolors[im.get_array()[j, i] > threshold])
            ax.text(i, j, text_matrices[j][i], **kw)
    #Font type and size of tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontproperties(ticks_font)
        tick.label.set_fontsize(12)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontproperties(ticks_font)
        tick.label.set_fontsize(12)
    #Orient the x labels to have an angle
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    #Save figure
    plt.savefig(output_file, dpi=400)

if __name__ == "__main__":
    generate_heatmap(both_matrix, output_file = 'plots/both_heatmap.png', cmap = "Blues")
    generate_heatmap(insert_matrix, output_file = 'plots/insert_heatmap.png', cmap = "Blues")
    generate_heatmap(swap_matrix, output_file = 'plots/swap_heatmap.png', cmap = "Blues")