import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(data, column, title="Distribution", figsize=(6, 4)):
    plt.figure(figsize=figsize)
    sns.countplot(x=column, data=data)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()