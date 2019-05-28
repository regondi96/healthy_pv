#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_trained = pd.read_csv("path.csv")
data_input = pd.read_csv("path.csv")


# In[3]:


data_trained.head()


# In[4]:


#Creando una nueva lista de missing values para que Pandas reconozca datos vacíos. 
missing_values = ["-999.00"]
data_input = pd.read_csv("/Users/hogar/Desktop/dataset.csv", na_values = missing_values)


# In[5]:


media1 = data_input["ALLSKY_SFC_SW_DWN"].median()
data_input["ALLSKY_SFC_SW_DWN"].fillna(media1, inplace=True)
media2 = data_input["KT_CLEAR"].median()
data_input["KT_CLEAR"].fillna(media2, inplace=True)
media3 = data_input["KT"].median()
data_input["KT"].fillna(media3, inplace=True)


# In[6]:


#Visualizando de nuevo la data
data_input["TS"].plot()


# In[7]:


data_trained["Temperature"].plot()


# In[8]:


type(data_trained)


# In[9]:


variables_t = ["Insolation", "KT_Index", "Temperature"]
variables_i = ["ALLSKY_SFC_SW_DWN", "KT", "TS"]


# In[10]:


array_trained = data_trained[variables_t]
array_input = data_input[variables_i]


# In[11]:


a_trained = array_trained.values
a_input1 = array_input.values


# In[12]:


a_input1.shape


# In[13]:


a_input = a_input1[0:1062]


# In[14]:


a_input[:,2]


# In[15]:


a_trained[:,2]


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.scatter(a_trained[:,2], a_input[:,2])
plt.show()


# In[17]:


sky_input = a_input[:,2]
sky_input


# In[18]:


median_sky = np.median(sky_input)
median_sky


# In[19]:


sky_trained = a_trained[:,2]
sky_trained


# In[20]:


median_trained = np.median(sky_trained)
median_trained


# In[21]:


r_mean = pearsonr(a_input[:,2], a_trained [:,2])
print (r_mean)


# In[48]:


x = sky_trained
y = sky_input
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(20, 12))
fig.subplots_adjust(hspace=0.5, left=0.1, right=1)
ax = axs[0]
hb = ax.hexbin(x, y, gridsize=70, cmap='viridis')
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Perfil de Temperatura")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('n-cantidad')

ax = axs[1]
hb = ax.hexbin(x, y, gridsize=70, bins='log', cmap='magma')
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Perfil Escalado")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

plt.show()


# In[ ]:




