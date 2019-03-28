import datetime
import itertools

import pandas as pd
import matplotlib.pyplot as plt

from topic_labeling import *

plt.style.use("seaborn")
df = pd.read_csv("hmdp_eval_save.tsv", sep="\t")
rows = [1,3]

marker = itertools.cycle(("o, ."))

fig, ax = plt.subplots()
labels = []

for i in rows:
    df[i]["comments"].date.value_counts().plot(kind="line", marker=next(marker))
    labels.append(" ".join(df[i]["label"]))

ax.legend(labels)

plt.axvline(datetime.datetime.strptime("2016-04-13", "%Y-%m-%d").date(), linestyle="--", color="grey")
plt.text(datetime.datetime.strptime("2016-04-13", "%Y-%m-%d").date(), ax.get_ylim()[1]-4,
         "Article about White Student Union", bbox=dict(facecolor='white', alpha=0.9))
plt.axvline(datetime.datetime.strptime("2016-04-26", "%Y-%m-%d").date(), linestyle="--", color="grey")
plt.text(datetime.datetime.strptime("2016-04-26", "%Y-%m-%d").date(), ax.get_ylim()[0]+90,
         "Amal Clooney critizes Trump", bbox=dict(facecolor='white', alpha=0.9))

plt.grid(b=True)
plt.show()