import numpy as np

#points = [(np.array([2]),4), (np.array([4]),2)]
#d = 1

# Generate artificial data
true_w = np.array([1, 2, 3, 4, 5])

d = len(true_w)
points = []
for i in range(10000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn(d) 
    # What is the effect of the noise when you generate
    #print(x, y)
    points.append((x,y))
    
#for i in points:
#	print(i)
#
#for i, j in points:
#	print('i,j=', i, j)

def F(w):
	return sum((w.dot(x)-y)**2 for x, y in points) / len(points)
#def F1(w):
#	return sum([(w*x-y)**2 for x, y in points])
#
#def F2(w):
#	total = 0
#	for x, y in points:
#		total += (w*x - y)**2
#	return total

def dF(w):
	return sum(2*(w.dot(x)-y)*x for x, y in points) / len(points)

def sF(w, i):
    x, y = points[i]
    return (w.dot(x)-y)**2

def sdF(w, i):
    x, y = points[i]
    return 2*(w.dot(x)-y)*x
    

# Gradient descent
#def Gradient_descent_hoan_created(w, eta, number_steps):
#	for t in range(number_steps + 1):
#		value = F(w)
#		gradient = dF(w)
#		w -= eta * gradient
#		print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))
#	return 'iteration {}: w = {}, F(w) = {}'.format(t, w, value)

def Gradient_descent_professor_teachs(F, dF, d):
	# Gradient descent
	w = np.zeros(d)
	eta = 0.01
	for t in range(1000):
		value = F(w)
		gradient = dF(w)
		w = w - eta * gradient
		print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))
        
#def StochasticGradient_descent_professor_teachs(sF, sdF, d, n):
#	# Gradient descent
#	w = np.zeros(d)
#	eta = 1
#    numUpdate = 0
#	for t in range(1000):
#        for i in range(n):
#    		value = sF(w, i)
#    		gradient = sdF(w, i)
#            numUpdates += 1
#            eta = 1.0 / numUpdates
#    		w = w - eta * gradient
#		print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))        

def Sto(sF, sdF, d, n):
    w = np.zeros(d)
    eta = 1
    numUpdate = 0
    for t in range(1000):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            numUpdates += 1
            eta = 1.0 / numUpdates
            w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))  
print(Gradient_descent_professor_teachs(F, dF, d))		
Sto(sF, sdF, d, len(points))
#print('F2(2):', F2(2))			
#print(Gradient_descent_hoan_created(0, 0.01, 100))




#print(F(2))	
#print('F1(2):', F1(2))
