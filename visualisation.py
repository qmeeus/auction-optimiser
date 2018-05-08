import matplotlib.pyplot as plt


def plot_gender_repartition(data, **kwargs):
    caption = kwargs["caption"]; del kwargs["caption"]
    gender_repartition = data.groupby([kwargs["index"], kwargs["columns"]], as_index=False).count().pivot(**kwargs)
    gender_repartition["Total"] = gender_repartition["Male"] + gender_repartition["Female"]
    gender_repartition["Female"] = gender_repartition["Female"] / gender_repartition["Total"] * 100
    gender_repartition["Male"] = gender_repartition["Male"] / gender_repartition["Total"] * 100
    gender_repartition[["Female", "Male"]].plot(kind="barh", stacked=True, legend=False)
    ax = plt.gca()
    ax.axvline(50, color="k", zorder=0)
    format_chart(ax, True)
    label_barh_chart(ax)
    plt.title("{}: Gender Repartition by {}".format(caption, kwargs["index"]), y=-.2)
    plt.legend(bbox_to_anchor=(.7, 1.1), ncol=2)
    plt.subplots_adjust(left=.16, bottom=.2)
    
def format_chart(ax, multiline_labels=False, ticklabel_size=10):
    [spine.set_visible(False) for spine in ax.spines.values()]
    #[tl.set_visible(False) for tl in ax.get_xticklabels()]
    ax.yaxis.set_label_text("")
    [tl.set(fontsize=ticklabel_size) for tl in ax.get_yticklabels()]
    if multiline_labels:
        ylabels = ax.get_yticklabels()
        new_labels = [label.get_text()[::-1].replace(" ", "\n", 1)[::-1] for label in ylabels]
        ax.set_yticklabels(new_labels)
        
def label_barh_chart(ax):
    text_settings = dict(fontsize=9, fontweight='bold', color="w")
    rects = ax.patches
    for i, rect in enumerate(rects):
        width = rect.get_width()
        x_pos = width / 2 if i in range(len(rects) // 2) else 100 - width / 2
        color = "pink" if i in range(len(rects) // 2) else "#2C6388"
        rect.set_facecolor(color)
        label = "{:.1f}%".format(rect.get_width())
        ax.text(x_pos, rect.get_y() + rect.get_height()/2, label, ha='center', va='center', **text_settings)
        
def label_barchart(ax):
    text_settings = dict(fontsize=9, fontweight='bold', color="w")
    rects = ax.patches
    for i, rect in enumerate(rects):
        x_pos = rect.get_x() + rect.get_width() / 2
        label = "{:.1%}".format(rect.get_height())
        ax.text(x_pos, .05, label, ha='center', va='center', **text_settings)
