import matplotlib.pyplot as plt

        
def label_barchart(ax):
    text_settings = dict(fontsize=9, fontweight='bold', color="Silver")
    rects = ax.patches
    for i, rect in enumerate(rects):
        x_pos = rect.get_x() + rect.get_width() / 2
        label = "{:.1%}".format(rect.get_height())
        ax.text(x_pos, .05, label, ha='center', va='center', **text_settings)
