# In[1]:


#Importing python packages
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal,GlorotNormal


# In[2]:


#Loading data. Can be downloaded or implemented from UCI Reposity(link in article)
dataframe = pd.read_csv("heart_failure_clinical_records_dataset.csv")
dataframe.head(5)


# In[3]:


#Dropping follow-up time and sex variables
dataframe.drop(['time','sex'],axis=1,inplace=True)


# In[4]:


#Checking data information 
dt = dataframe
dt.info()


# In[5]:


#Splitting data under random train_test_split with proportions 6:4
from sklearn.model_selection import train_test_split
whole = dt.iloc[:,0:11]
train, test = train_test_split(whole,test_size=0.4, random_state=123123,shuffle=True)


# In[6]:


#Checking imbalance in target variable: mortality in heart failure patients
np.bincount(train['DEATH_EVENT'])


# In[7]:


#Normalizing Data for convenience
mn1 = MinMaxScaler()
train_ = mn1.fit_transform(train)
train_ = pd.DataFrame(train_, columns=dt.columns)


# In[8]:


#Constructing GAE_AM model
tf.keras.utils.set_random_seed(321321)

class GAE_AM(tf.keras.Model):
    def __init__(self, input_dim,lat_dim):
        super(GAE_AM, self).__init__()
        self.input_dim = input_dim
        self.lat_dim = lat_dim
        h1 = input_dim*8
        h2 = input_dim*4
        self.h1 = h1
        self.h2 = h2
        mat1 = np.zeros(input_dim*h1).reshape(input_dim,h1)
        for i in np.arange(0,input_dim,1):
            for j in np.arange(0,int(h1/input_dim),1):
                mat1[i,(j+(i*int(h1/input_dim)))] = 1
        
        mask1 = np.array(mat1).reshape(input_dim,h1)
        class mask_1(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask1*w
        
        
        mat2 = np.zeros(h1*h2).reshape(h1,h2)
        for i in np.arange(0,input_dim,1):
            mat2[(i*int(h1/input_dim)):((i+1)*int(h1/input_dim)),(i*int(h2/input_dim)):((i+1)*int(h2/input_dim))] = 1
        
        mask2 = np.array(mat2).reshape(h1,h2)
        class mask_2(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask2*w    
            
        mat3 = np.zeros(h2*lat_dim).reshape(h2,lat_dim)
        for i in np.arange(0,lat_dim,1):
            mat3[(i*int(h2/lat_dim)):((i+1)*int(h2/lat_dim)),i] = 1
        
        mask3 = np.array(mat3).reshape(h2,lat_dim)
        class mask_3(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask3*w
        
        mat4 = np.zeros(lat_dim*h2).reshape(lat_dim,h2)
        for i in np.arange(0,lat_dim,1):
            mat4[i,(i*int(h2/lat_dim)):((i+1)*int(h2/lat_dim))] = 1
        
        mask4 = np.array(mat4).reshape(lat_dim,h2)
        class mask_4(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask4*w

        mat5 = np.zeros(h2*h1).reshape(h2,h1)
        for i in np.arange(0,input_dim,1):
            mat5[(i*int(h2/input_dim)):((i+1)*int(h2/input_dim)),(i*int(h1/input_dim)):((i+1)*int(h1/input_dim))] = 1
        
        mask5 = np.array(mat5).reshape(h2,h1)
        class mask_5(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask5*w
            
        mat6 = np.zeros(h1*input_dim).reshape(h1,input_dim)
        for i in np.arange(0,input_dim,1):
            mat6[(i*int(h1/input_dim)):((i+1)*int(h1/input_dim)),i] = 1
        
        mask6 = np.array(mat6).reshape(h1,input_dim)
        class mask_6(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask6*w
        
        mat_a = np.ones(lat_dim*lat_dim).reshape(lat_dim,-1)
        mat_a[lat_dim-1,:] = 0
        
        mask__a = np.array(mat_a).reshape(lat_dim,-1)
        class mask_a(tf.keras.constraints.Constraint):
            def __call__(self,w):
                return mask__a*w
        
        
        
        
        k = tf.keras.initializers.HeNormal(123)
        self.Enc = Sequential([ #h1,h2
            Dense(units = h1, use_bias=False, input_shape = (input_dim, ),kernel_initializer=k,kernel_constraint=mask_1()),
            Dense(units = h2, use_bias=False,kernel_initializer=k,kernel_constraint=mask_2()),
            Dense(units=lat_dim,use_bias=False,kernel_initializer=k,activation="linear",kernel_constraint=mask_3()) 
        ])
        
        self.Dec = Sequential([
            Dense(units=lat_dim,use_bias=False,input_shape = (lat_dim, ),kernel_initializer=k,kernel_constraint=mask_a()),
            Dense(units = h2, use_bias=False, kernel_initializer=k,kernel_constraint=mask_4()),
            Dense(units = h1, use_bias=False,kernel_initializer=k,kernel_constraint=mask_5()),
            Dense(units=input_dim,use_bias=False,kernel_initializer=k,activation="sigmoid",kernel_constraint=mask_6())
        ])
        
        
    
        
    def enc(self,x):
        z = self.Enc(x, training=True)
        return z
        
            
    def dec(self,z_):
        x_hat = self.Dec(z_, training=True)
        return x_hat
        
    mse_loss = tf.keras.losses.MeanSquaredError()
    def ae_loss(model, x,input_dim,alpha,binary, non_b):
        z_1 = model.enc(x)
        x_hat = model.dec(z_1)
        ad = 0
        ad2 = 0
        for i in non_b:
            ad +=mse_loss(x[:,i], x_hat[:,i])
        for j in binary:
            ad2 += mse_loss(x[:,j],x_hat[:,j])
        return (1-alpha)*ad/len(non_b)+alpha*ad2/len(binary)


# In[9]:


#Classification of binary/continous variables from data info
b_1 = [1,3,5,9,10]
b_2 = [0,2,4,6,7,8]


# In[28]:


#GAE_AM fitting procedure

Epochs = 800
tf.keras.utils.set_random_seed(321321)
lat_dim_ = 11
input_dim_ = 11
GC = GAE_AM(input_dim_,lat_dim_)
mse_loss = tf.keras.losses.MeanSquaredError()

i = 0
loss_of_ae = []
loss_of_cp = []
alpha = 0.2
rho = 0.1 
gamma=0.9 
beta = 1.01 
lamb = 1.0 #L1-regularization
lamb2 = 2.0 #L1-for target row

enc_opt = tf.keras.optimizers.Adam(learning_rate=0.008)
dec_opt = tf.keras.optimizers.Adam(learning_rate=0.008)

while i < Epochs:
    ae_in = np.array(train_)
    with tf.GradientTape() as enc_t, tf.GradientTape() as dec_t:
        ae_l = GC.ae_loss(ae_in,input_dim_,alpha=0.3, binary=b_1,non_b=b_2)
        h_a = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(GC.Dec.weights[0], GC.Dec.weights[0])))-lat_dim_
        cs_l = ae_l+alpha*h_a+rho*0.5*tf.math.abs(h_a)**2+lamb*tf.norm(GC.Dec.weights[0], ord=1, axis=[-2,-1])
    loss_of_ae.append(ae_l)
    loss_of_cp.append(cs_l)
    grad_enc = enc_t.gradient(cs_l, GC.Enc.trainable_variables)
    grad_dec = dec_t.gradient(cs_l, GC.Dec.trainable_variables)
    enc_opt.apply_gradients(zip(grad_enc, GC.Enc.trainable_variables))
    dec_opt.apply_gradients(zip(grad_dec, GC.Dec.trainable_variables))
    h_a_new = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(GC.Dec.weights[0], GC.Dec.weights[0])))-lat_dim_
    alpha =  alpha + rho * h_a_new
    if (tf.math.abs(h_a_new) >= gamma*tf.math.abs(h_a)):
        rho = beta*rho
    else:
        rho = rho
    if (i+1) %20 == 0: print(i+1,ae_l,cs_l)
    i = i+1


# In[29]:


print(loss_of_cp[-1])
print(loss_of_ae[-1])


# In[30]:


#Checking Causal Loss values per epoch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cp, color="red",label="causal_loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


# In[31]:


#Checking AutoEncoder Loss values per epoch
plt.plot(loss_of_ae, color="darkblue",label="ae_loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


# In[32]:


#Resulted Weighted Adjacency matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(np.array(GC.Dec.weights[0]), cmap="vlag",center=0)
plt.show()




# In[33]:


filtered = np.array(tf.where(tf.greater(np.abs(np.array(GC.Dec.weights[0])), np.quantile(np.abs(np.array(GC.Dec.weights[0])),0.85),0),1,0))


# In[34]:


#Returned binary adjacency matrix
sns.heatmap(filtered, cmap="vlag",center=0)
plt.show()


# In[35]:


#Eliminating minor acyclicity errors
filtered_ = filtered*1
filtered_[4,0] = filtered_[4,9] = filtered_[5,6] = 0
filtered_


# In[36]:


#Final binary adjacency matrix
sns.heatmap(filtered_,cmap="vlag",center=0)
plt.show()


# In[37]:


#Visualizing final adjacency in networkx
import networkx as nx
np.random.seed(100)
A = filtered_*1
G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
nx.draw(G,with_labels=True,connectionstyle="arc, rad=1",node_size=0.5e+3,font_size=15,node_color="gray") #12,3,10,11,2,6,0,4,11,


# In[ ]:




