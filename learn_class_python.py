''' I want to test if some methods/functions are called.
    I must to install test Double.
    I think that a method/function is a object
    So I use another method/function for double
    And install the double is very simple with assign statement
    Dynamic language make it easier to install double
'''

class baseclass():
    def a(self):
        self.b()
    def b(self):
        print("in base")
    def c(self):
        print("in base.c")
class drivedclass(baseclass):
    def b(self):
        self.c()
dc = drivedclass()
dc.a()
