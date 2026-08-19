

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def factorixze(x, maxi, mini ):
    #  ax  + b = y  || x ==0 ==> y = mini || x == 1 ==> y = maxi
    b = mini 
    a = maxi - b 
    return a * x + b 


