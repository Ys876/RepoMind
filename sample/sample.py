# fake functions to be parsed by tree-sitter
#blehbleh
def purdue():
    print("boilerup")
def login(user, pwd):
    if ((user == "pete") and (pwd == "boilerUp123")):
        print("success")
    else:
        print("fail")

def get_user():
    return "pete"

def get_pwd():
    return "boilerUp123"