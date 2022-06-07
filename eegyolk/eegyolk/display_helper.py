""" A .py file to help with displaying information.   """

import matplotlib.pyplot as plt


def show_plot(x = False, y = False, title = "", xlabel = "", ylabel = "", legend = False, show = True):
    '''
     Show plot with title and lables in 1 line.

    Args:
     x: 1D numpy array        
        
     y: 1D numpy array

    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)
    if x and y:
        plt.plot(x, y)
    if show:
        plt.show()


def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


color_dictionary = {
    1: "#8b0000",
    2: "#008000",
    3: "#000080",
    4: "#ff0000",
    5: "#ff1493",
    6: "#911eb4",
    7: "#87cefa",
    8: "#ffd700",
    9: "#696969",
    10: "#000000", 
    11: "#1e90ff",
    12: "#7fff00",
}