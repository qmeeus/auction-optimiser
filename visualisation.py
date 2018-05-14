import matplotlib.pyplot as plt

        
def label_barchart(ax, fmt="{:.1%}"):
    text_settings = dict(fontsize=12, fontweight='bold', color="White")
    rects = ax.patches
    for i, rect in enumerate(rects):
        x_pos = rect.get_x() + rect.get_width() / 2
        label = fmt.format(rect.get_height())
        ax.text(x_pos, .5, label, ha='center', va='center', **text_settings)
