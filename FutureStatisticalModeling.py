singlet_rect = 9368 
double_rect = 1571
triple_rect = 316 
quad_rect = 78

#Independent events
DiProp = double_rect/singlet_rect #0.16769
ExpTripNum = singlet_rect*DiProp*DiProp+double_rect*DiProp #526.9088
#Simplification of Expected Triple That is equivalent
#SimpleExpTripNum = 2*double_rect*DiProp

ExpQuadNum = singlet_rect*DiProp*DiProp*DiProp+double_rect*DiProp*DiProp+triple_rect*DiProp #141.3545
#Simplification of Expected Quad That is equivalent
#SimpleExpQuadNum = 2*double_rect*DiProp*DiProp+triple_rect*DiProp



# What proportion is needed to recreate the true values
TrueProp = .119
ExpTripNum2 = singlet_rect*TrueProp*TrueProp+double_rect*TrueProp#319.609248
ExpQuadNum2 = singlet_rect*TrueProp*TrueProp*TrueProp+double_rect*TrueProp*TrueProp+triple_rect*TrueProp#75.6375



#Dependent Events
#Take this as if there are only 4 neighbors to go to
DiProp = double_rect/singlet_rect #0.16769
ExpTwoDiNum = singlet_rect*DiProp*DiProp*3/4
ExpDoubleDiNum = double_rect*DiProp*3/4
ExpTripNum = ExpTwoDiNum+ExpDoubleDiNum
print('ExpTripNum:',ExpTripNum)




ExpThreeDiNum = ExpTwoDiNum*DiProp*2/4



ExpDoubleTwoDiNum = double_rect*DiProp*3/4
ExpTripDiNum = ExpThreeDiNum+ExpDoubleDiNum
ExpQuadNum = ExpThreeDiNum+ExpDoubleTwoDiNum+ExpTripDiNum
print('ExpQuadNum:',ExpQuadNum)






