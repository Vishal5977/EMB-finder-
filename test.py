import pyembroidery
import os

dst_file = r'C:\Users\Asus\all pendrive design\dual hybrid\Krishna offer design\2383\FROUNT.DST'

pattern = pyembroidery.read(dst_file)
out_file = r'C:\Users\Asus\embroidery-finder\test_output.png'
pyembroidery.write(pattern, out_file)

if os.path.exists(out_file):
    print('SUCCESS - DST converted to image!')
    print('Saved at:', out_file)
else:
    print('FAILED - something went wrong')