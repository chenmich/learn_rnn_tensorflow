''' I want to test if some methods/functions are called.
    I must to install test Double.
    I think that a method/function is a object
    So I use another method/function for double
    And install the double is very simple with assign statement
    Dynamic language make it easier to install double
'''

def replaceFunction(obj,y):
    obj.someFunctionCalled = True


class some():
    def someFunction(self, x):
        print(x)

#before test double is installed
someInstance = some()
someInstance.someFunction('x')

#install test double
someInstance.someFunctionCalled = False
someInstance.someFunction=replaceFunction

#after test double is installed
someInstance.someFunction(someInstance, 'y')
assert someInstance.someFunctionCalled is not False


