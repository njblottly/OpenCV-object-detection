_ = 999

value = 42

match value:
    case 10:
        print("ten")
    case _:
        print("default case")
        print(_)

print(_)

