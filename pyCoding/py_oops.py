# -*- coding: utf-8 -*-
"""
@author: s.jayanthi
"""

# Class attributes vs instance attributes
class classA(object):
    count = 0;
    garbage_collect = [];
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                classA.garbage_collect.append(val);
            else:
                setattr(self,key,val);
        classA.count+=1;
        self._var1 = 'private variable 1';
        self.__var2 = 'private variable 2';
classA_obj1 = classA();
print(classA_obj1.__class__.count)
print(classA_obj1.__class__.garbage_collect)
print(classA_obj1.__dict__)
classA_obj2 = classA();
print(classA_obj2.__class__.count)
print(classA_obj2.__class__.garbage_collect)
print(classA_obj2.__dict__)
classA_obj3 = classA(lst=['hello'],dct={'2':'two'},it=34,strg='Lastly');
print(classA.count)
print(classA.garbage_collect)
print(classA_obj3.__dict__)
classA_obj4 = classA(lst=['bye'],dct={'3':'three'},it=43,strg='Firstly');
print(classA.count)
print(classA.garbage_collect)
print(classA_obj4.__dict__)

# Whats with underscores
"""
._variable is semiprivate and meant just for convention to developer
.__variable imples you are just name mangling i.e adding _<className> in front of __variable or __method name. Helps in avoiding accidental access.
.__variable__ is typically reserved for builtin methods or variables

Access ._variable normally as <InstanceName>.<variable name with leading single underscore> Ex: classA_obj1._var1
Access .__variable by typing <InstanceName>._<ClassName><variable name with leading double underscores> Ex: classA_obj1._ClassA__var2
"""

# OOPS
"""
# Main Topics
1. Inheritance; Single-Level and Multiple-Level
2. Encapsulation
3. Polymorphism

# Some Inferences
1. Using super(<className>,self).__init()__, you are only calling one level at once. Unless the next level has another super in its init, the
second level classes wont be initialized.
"""
# Example of multiple (same-level) inheritances
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
# https://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
class BaseClassA(object):
    def __init__(self, a1, a2, *args):
        self.a1 = a1;
        self.a2 = a2;
        self.__var1 = 10;
        self._var2 = 15;
        self.var3 = 20;
        print('a1:{}, a2:{}'.format(self.a1,self.a2))
        super().__init__(*args);
    def __method1(self):
        return 100;
    def _method2(self):
        return 150;
    def method3(self):
        return 200;
    def change_a1(self):
        print('a1 Before:{}'.format(self.a1)); self.a1**=2; print('a1 After:{}'.format(self.a1))
    def change_a2(self):
        print('a2 Before:{}'.format(self.a2)); self.a2**=2; print('a2 After:{}'.format(self.a2))
class BaseClassE(object):
    def __init__(self,e):
        self.e = 10;
        print('e: ', self.e)
class BaseClassB(BaseClassE):
    def __init__(self, b, *args_left):
        self.b = b;
        self.__var1 = 13;
        self._var2 = 18;
        self.var3 = 23;
        print('b: ', self.b)
        super().__init__(*args_left)
    def __method1(self):
        return 130;
    def _method2(self):
        return 180;
    def method3(self):
        return 230;
    def change_b(self):
        print('b Before:{}'.format(self.b)); self.b**=2; print('b After:{}'.format(self.b))
class BaseClassC(BaseClassA, BaseClassB):
    def __init__(self, a1, a2, b, c, e):
        self.c = c;
        print('c: ',c)
        super().__init__(a1,a2,b,e);
        print(self.__dict__);
    def method3(self):
        return 2020;
class WrapperClassD(BaseClassC):
    def __init__(self, a, b):
        super().__init__(a, b);
        print(self.__dict__);


base_class_c = BaseClassC(1010,1009,1008,1007,1006); # See for same name variables, the latest base class that initialized it is the winner
print(base_class_c.method3()); # See for same name methods, the latest method in execution tree is the winner
print(base_class_c._method2());
print(base_class_c.change_b());
print(dir(base_class_c));