#!/usr/bin/python3

import sys

import datetime

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler 

debug = True

# Making a date converter 

month_conv = { \
	"jan" : 1 ,\
	"fev" : 2 ,\
	"mar" : 3 ,\
	"abr" : 4 ,\
	"mai" : 5 ,\
	"jun" : 6 ,\
	"jul" : 7 ,\
	"ago" : 8 ,\
	"set" : 9 ,\
	"out" : 10 ,\
	"nov" : 11, \
	"dez" : 12 }
	

X = []
Xheader = {}

with open("X.csv", "r") as Xfile:

	# First line is header
	
	header_line = Xfile.readline();
	
	header = [ h.strip() for h in header_line.split(";") ]
	
	# This is to check if all data lines have the same length
	# Firs column is line tag
	number_of_columns = len(header)
	
	if debug:	
		print(header)			
	
	for line in Xfile.readlines():
	
		data = [ d.strip().replace(".","").replace(",",".") for d in line.split(";") ]
		
		if debug:	
			print(data)
			
		if len(data) != len(header):
			print("Error! Data line length mismatch!")
			print("Expected ", len(header), ", got ", len(data))
		
			sys.exit(-1)
			
		X.append( data )					

Y = [] 
Yheader = {}
			
with open("Y.csv", "r") as Yfile:

	header_line = Yfile.readline();
	
	header = [ h.strip() for h in header_line.split(";") ]
	
	# This is to check if all data lines have the same length
	# Firs column is line tag
	number_of_columns = len(header)	

	for line in Yfile.readlines():
	
		data = [ d.strip().replace(".", "") for d in line.split(";") ]
		
		if debug:	
			print(data)
			
		if len(data) != len(header):
			print("Error! Data line length mismatch!")
			print("Expected ", len(header), ", got ", len(data))
		
			sys.exit(-1)
			
		Y.append( data )
	
if len(X) != len(Y):

	print("Error! training examples length does not match output length!")

	sys.exit(-1)	

	
# Removing data before dez/10
date_limit = datetime.datetime(2010, 12, 1)

X_train = []
Y_train = []

X_test = []
Y_test = []
index = 0
while True: 

	if index >= len( X ):
		break
	
	str_date = X[ index ][ 0 ] .split("/")
	x_month = str_date[ 0 ]
	x_year = str_date[ 1 ]
	x_date = datetime.datetime( int(x_year) + 2000 , int( month_conv[ x_month ] ) , 1)

	str_date = Y[ index ][ 0 ] .split("/")
	y_month = str_date[ 0 ]
	y_year = str_date[ 1 ]		
	y_date = datetime.datetime( int(y_year) + 2000 , int( month_conv[ y_month ] ) , 1)
	
	if debug:
		print( "[", index, "]: ", x_date, " | ", y_date )
	
	if x_date != y_date:
	
		print("Error! X date is not the same as Y date!")	
		sys.exit(-1)

	if x_date <= date_limit:
	
		print("Removing ... ", x_date)
	
		del X[ index ]
		del Y[ index ]
				
	else: 		
		# o teste serah prever o consumo no ano de 2023
		if x_date < datetime.datetime(2023, 1, 1):
		
			X_train.extend( [ X[ index ] ] )
			Y_train.extend( [ Y[ index ] ] )
			
		else: 
		
			X_test.extend( [ X[ index ] ] )
			Y_test.extend( [ Y[ index ] ] )	
	
		index += 1
	

with open("test.csv", "w") as test_file:

	for index in range( len(X) ):
	
		test_file.write( ";".join( X[index] ) + ";" + Y[index][1] ) 
		
		if index <= len(X) - 1:
		
			test_file.write( "\n" )
	

# Removing date tags
X_train = [ x[ 1: ] for x in X_train ]
Y_train = [ y[ 1 ] for y in Y_train ]

X_test = [ x[ 1: ] for x in X_test ]
Y_test = [ y[ 1 ] for y in Y_test ]

for index in range( len(X_train) ):

	X_train[ index ] = [ float(d) for d in X_train[ index ] ]
		
	Y_train[ index ] = float( Y_train[ index ] )
	
for index in range( len(X_test) ):	

	X_test[ index ] = [ float(d) for d in X_test[ index ] ]
		
	Y_test[ index ] = float( Y_test[ index ] )

# print( Y_train )

				
scalerX = StandardScaler()
scalerX.fit( X_train )

scaled_X_train = scalerX.transform( X_train )
scaled_X_test = scalerX.transform( X_test )

scalerY = StandardScaler()
scalerY.fit( [ [y] for y in Y_train ] )

scaled_Y_train = scalerY.transform( [ [y] for y in Y_train ] )
scaled_Y_test = scalerY.transform( [ [y] for y in Y_test ] )

print( scaled_Y_test )

hidden_layer_neurons = len( X_train[ 0 ] )

EnergyPredictor = MLPRegressor( \
	activation = "tanh" ,\
	solver = "adam" ,\
	max_iter = 5000 ,\
	hidden_layer_sizes = [ hidden_layer_neurons for x in range(11) ] ,\
	learning_rate_init = 0.0001,
	random_state = 1,
	verbose = True )
	
EnergyPredictor.fit( scaled_X_train, [ y[0] for y in scaled_Y_train ] )

with open("coefs.dat","w") as coefs:
	coefs.write ( str( EnergyPredictor.coefs_ ) )
	
	
print( len( EnergyPredictor.coefs_ ) , len( EnergyPredictor.coefs_[0] )  )

MAPE = 0

for test in range( len( scaled_X_test ) ):

	Y_scaled_pred = EnergyPredictor.predict( [ scaled_X_test[test] ] )
	
	Y_real = scalerY.inverse_transform( [ scaled_Y_test[test] ] )
	Y_pred = scalerY.inverse_transform( [ Y_scaled_pred ] )
	
	print( Y_real , Y_pred, Y_pred[0] / Y_real[0] )
	
	MAPE += abs( Y_pred[0] - Y_real[0] ) / Y_real[0]
	
	print("\n")
	
MAPE /= len( scaled_X_test )
MAPE *= 100

print("MAPE :", MAPE )
	
print( EnergyPredictor.score( scaled_X_test, scaled_Y_test ) )	


		

	
