class Person:
    def __call__(self, name):
        print("__call__" + " Hello " + name)

    def hello(self, name):
        print("Hello " + name)


person = Person()
person("zhang")
person.hello("lisi")
person()
