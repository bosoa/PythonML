import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('convenient_store.csv')
data.hourly_wage.hist(bins=10)
plt.show()
