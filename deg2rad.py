import math as m

# functions
def deg2rad(deg: float) -> float:
    return deg * m.pi / 180.0

def rad2deg(rad: float) -> float:
    return rad * 180.0 / m.pi

# main
while True:
    responce = input('deg or rad? (d/r/q): ')
    if responce == 'q':
        break
    if responce == 'd':
        print('rad: ' + str(deg2rad(float(input('deg: ')))) + '\n')
    elif responce == 'r':
        print('deg: ' + str(rad2deg(float(input('rad: ')))) + '\n')
    else:
        print('Invalid input')
