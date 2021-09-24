from ij import IJ,ImagePlus,ImageStack
import os, itertools
import glob


data_name = 'XXXXsample' # directory of samples
pwd_dpath = 'C:/XXX/YYY/ZZZ/' # please specify the current directory
data_dpath = os.path.join(pwd_dpath, 'results', data_name, 'gauss')
out_dpath = os.path.join(pwd_dpath, 'results', data_name, 'thin')

img_fpath_list = [os.path.join(data_dpath, d) for d in os.listdir(data_dpath)]
print(img_fpath_list)
angle = 36
min_length = 40

def main():
	for img_fpath in img_fpath_list:
	    print(img_fpath)
	    img_name = img_fpath.rsplit('\\', 1)[-1]
	    imp = IJ.openImage(img_fpath)
		
	    imp2 = IJ.run(imp,"Lpx Filter2d", "filter=lineFilters__ fftswapquadrants linemode=lineByRotWs__ numangles={0} minlen={1}".format(angle, min_length))
	    IJ.run('8-bit')
	    IJ.saveAs(imp2,'tif', os.path.join(out_dpath, img_name))
	    IJ.run('Close')

if __name__ == '__builtin__':
	main()
	print('Complete')
