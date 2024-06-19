import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial

class FollowDotCursor(object):

    def __init__(self, ax, x, y, L, AF, y2):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        y = y[np.abs(y - y.mean()) <= 3 * y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.ax = ax
        self.fig = ax.figure
        self.L = L
        self.AF = AF
        self.y2 = y2
        self.annotation = {}
        self.dot = ax.scatter([x.min()], [y.min()], s=60, color='green', alpha=0.7)
        self.dot2 = ax.scatter([x.min()], [y2.min()], s=60, color='green', alpha=0.7)

        self.xoffset, self.yoffset = -30, 30

        # self.annotation = ax.annotate("",xy=(x, y), xytext=(self.xoffset, self.yoffset),textcoords='offset points', ha='left', va='center',bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        self.annotation = ax.annotate("", xy=(x, y), xytext=(self.xoffset, self.yoffset), textcoords='offset points',
                                      ha='left', va='center',
                                      bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        self.annotation.set_visible(False)

        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def Get_closest_id(L, value):
        return list(L).index(min(L, key=lambda x: abs(x - value)))

    def __call__(self, event):
        ax = self.ax
        l = self.L
        af = self.AF
        y2 = self.y2
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        x, y = self.snap(x, y)
        self.annotation.xy = x, y
        id = list(l).index(min(l, key=lambda a: abs(a - x)))
        self.annotation.set_text("Index: " + str(id) + "\nLevel: " + str(af[id]))
        self.annotation.set_visible(True)
        event.canvas.draw()

        yy = y2[id]

        self.dot.set_offsets(np.column_stack((x, y)))
        self.dot2.set_offsets(np.column_stack((x, yy)))
        self.fig.canvas.draw_idle()

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.formatter, xy=(0, 0), ha='left',
                                 xytext=self.offsets, textcoords='offset points', va='center',
                                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                                 )
        annotation.set_visible(False)
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]
