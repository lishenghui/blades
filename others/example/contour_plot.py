import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch


class AnnotationHandler(HandlerLine2D):
    def __init__(self, ms, *args, **kwargs):
        self.ms = ms
        HandlerLine2D.__init__(self, *args, **kwargs)
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = ((height - ydescent) / 2.) * np.ones(len(xdata), float)
        legline = FancyArrowPatch(posA=(xdata[0], ydata[0]),
                                  posB=(xdata[-1], ydata[-1]),
                                  mutation_scale=self.ms,
                                  **orig_handle.arrowprops)
        legline.set_transform(trans)
        return legline,


fig, ax = plt.subplots()
ax.axis([-1, 6, 0, 3])
ax.plot([1.5, 1.5], label="plot")
# create annotations in the axes
annotate = ax.annotate('', xy=(0, 1), xytext=(5, 1),
                       arrowprops={'arrowstyle': '|-|'}, label="endline")
annotate2 = ax.annotate('', xy=(1, 2), xytext=(3, 2),
                        arrowprops=dict(arrowstyle='->', color="crimson"), label="arrow")
# create legend for annotations
h, l = ax.get_legend_handles_labels()
ax.legend(handles=h + [annotate, annotate2],
          handler_map={type(annotate): AnnotationHandler(5)})

plt.show()
