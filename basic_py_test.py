# ok this works with pyinstaller
import os
if __name__ == '__main__':
    
    with open('/Users/jonathan/Documents/Research/test.txt', 'w') as f:
        f.write(os.getcwd())