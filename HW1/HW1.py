
# Writen by Nguyen Van Chuong, 20161199
# Gwangju Institute of Science and Technology
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# generate N random points
#def generate(N,N_test):
N=100
N_test=50
X_n=np.random.rand(N,1)
t_n= np.sin(np.pi*2*X_n)+ np.random.randn(N,1)
X_test= np.random.rand(N_test,1)
y_test= np.sin(np.pi*2*X_test)+ np.random.randn(N_test,1)


def fitting_the_model(M):
    if M>=1:
        poly_features=PolynomialFeatures(degree=M, include_bias=False)
        X_train_poly=poly_features.fit_transform(X_n) # contain original X and its new features
        # Train the model
        model=LinearRegression()
        model.fit(X_train_poly,t_n) # Fit the model
        #predict of output training data
        y_train_predict=model.predict(X_train_poly) 
        # Plot
        X_plot=np.linspace(0,1,100).reshape(-1,1)
        X_plot_poly=poly_features.fit_transform(X_plot)
        y_plot_predict=model.predict(X_plot_poly)# just for plot
        plt.plot(X_n,t_n,"b.")
        plt.plot(X_plot,y_plot_predict,'-r')
        plt.show()
        print (np.sqrt(mean_squared_error(t_n, y_train_predict)))
        
    plt.plot(X_n,t_n,'b.')
    plt.plot(X_n,np.ones_like(X_n)*t_n.mean(),'-r')
    plt.show()
    print (np.sqrt(mean_squared_error(t_n, np.ones_like(X_n)*t_n.mean())))
    
def RMSerror_train(M):
    if M>=1:  
        poly_features=PolynomialFeatures(degree=M, include_bias=False)
        X_train_poly=poly_features.fit_transform(X_n) 
        # Train the model
        model=LinearRegression()
        model.fit(X_train_poly,t_n)
        # Evaluate on train data
        y_train_predict=model.predict(X_train_poly)
        return (np.sqrt(mean_squared_error(t_n,y_train_predict)))  
    return np.sqrt(mean_squared_error(t_n,np.ones_like(X_n)*t_n.mean()))


def RMSerror_test(M):
    if M>=1 :
        poly_features=PolynomialFeatures(degree=M, include_bias=False)
        X_train_poly=poly_features.fit_transform(X_n) 
        X_test_poly=poly_features.fit_transform(X_test)
       # Train the model
        model=LinearRegression()
        model.fit(X_train_poly,t_n)
       # Evaluate on test data
        y_test_predict=model.predict(X_test_poly)
        return (np.sqrt(mean_squared_error(y_test,y_test_predict)))
    return np.sqrt(mean_squared_error(y_test,np.ones_like(X_test)*y_test.mean()))



X_RMS_display=[0,1,2,3,4,5,6,7,8]
RMS_train_display=[RMSerror_train(0), RMSerror_train(1),RMSerror_train(2),RMSerror_train(3),RMSerror_train(4),RMSerror_train(5),RMSerror_train(6),RMSerror_train(7),RMSerror_train(8)]
RMS_test_display=[RMSerror_test(0),RMSerror_test(1),RMSerror_test(2),RMSerror_test(3),RMSerror_test(4),RMSerror_test(5),RMSerror_test(6),RMSerror_test(7),RMSerror_test(8)]
plt.plot(X_RMS_display,RMS_train_display,"-b",linewidth=2,label= "RMS_train")
plt.plot(X_RMS_display,RMS_test_display,"-r",linewidth=2,label= "RMS_test")
plt.show()




