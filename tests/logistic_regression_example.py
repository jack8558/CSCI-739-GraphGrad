from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn import datasets
import os
import sys
import numpy as np

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import graphgrad as gg


class CustomLogisticRegression():
    def __init__( self, learning_rate, iterations ) :         
        self.learning_rate = gg.tensor(learning_rate)         
        self.iterations = iterations 
          
    # Function for model training     
    def fit( self, X, Y ) :         
        # no_of_training_examples, no_of_features         
        self.m, self.n = list(X.dims())         
        # weight initialization         
        self.W = gg.zeros( [self.n] )         
        self.b = gg.tensor([0])        
        self.X = X         
        self.Y = Y 
          
        # gradient descent learning 
                  
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
      
    # Helper function to update weights in gradient descent 
      
    def update_weights( self ):            
        A = gg.tensor([1]) / ( gg.tensor([1]) + gg.exp( - ( self.W.matmul( self.X.transpose(0, 1) ) + self.b ) ) ) 
          
        # calculate gradients  
        tmp = ( A - self.Y )         
        tmp = gg.reshape( tmp, [self.m] )       
        dW = gg.matmul( tmp, self.X ) / gg.tensor(self.m)          
        db = gg.sum( tmp ) / gg.tensor(self.m)  
          
        # update weights     
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db 
          
        return self
      
    # Hypothetical function  h( x )  
      
    def predict( self, X ):    
        Z = gg.tensor([1]) / ( gg.tensor([1]) + gg.exp( - ( self.W.matmul( X.transpose(0, 1) ) + self.b ) ) )   
        Y = np.where( np.array(Z.to_list()) > 0.5, 1, 0 )         
        return Y 
    
def main(): 
      
    # Importing dataset     
    X, Y = datasets.load_breast_cancer(return_X_y=True) 
      
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/3, random_state = 0 ) 
      
    # Model training     
    model = CustomLogisticRegression( learning_rate = 0.1, iterations = 10000 ) 
      
    model.fit( gg.tensor(X_train), gg.tensor(Y_train) )    
    model1 = LogisticRegression(max_iter=1000)     
    model1.fit( X_train, Y_train) 
      
    # Prediction on test set 
    Y_pred = model.predict( gg.tensor(X_test) ) 
    Y_pred1 = model1.predict( X_test ) 
      
    # measure performance     
    correctly_classified = 0    
    correctly_classified1 = 0
      
    # counter     
    count = 0    
    for count in range( np.size( Y_pred ) ) :   
        
        if Y_test[count] == Y_pred[count] :             
            correctly_classified = correctly_classified + 1
          
        if Y_test[count] == Y_pred1[count] :             
            correctly_classified1 = correctly_classified1 + 1
              
        count = count + 1
          
    print( "Accuracy on test set by our model       :  ", (  
      correctly_classified / count ) * 100 ) 
    print( "Accuracy on test set by sklearn model   :  ", (  
      correctly_classified1 / count ) * 100 ) 
  
  
if __name__ == "__main__" :      
    main()
