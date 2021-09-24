import cv2
import numpy as np
import copy
import os, sys, itertools, argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--flag', type=str, help='Please select process type (filter, label, or eval)')
parser.add_argument('--name', type=str, help='Name of samples')
args = parser.parse_args()
data_name = args.name

data_dpath = './data/{}'.format(data_name)
out_dpath = './results/{}'.format(data_name)

membrane_img_fpath_list = os.listdir('{}/membrane'.format(data_dpath))
membrane_img_fpath_list = [os.path.join(data_dpath, 'membrane', d) for d in sorted(membrane_img_fpath_list)]
tj_img_fpath_list = os.listdir('{}\\tj'.format(data_dpath))
tj_img_fpath_list = [os.path.join(data_dpath, 'tj', d) for d in sorted(tj_img_fpath_list)]

pix_ths = [100]

if os.path.exists(out_dpath) is False:
    os.makedirs(out_dpath)
    os.makedirs('{}\\gauss'.format(out_dpath))
    os.makedirs('{}\\th'.format(out_dpath))
    os.makedirs('{}\\thin'.format(out_dpath))
    os.makedirs('{}\\label'.format(out_dpath))


def get_labeled_image(img, pix_ths=[100]):
    labeled_imgs = []
    for pix_th in pix_ths:
        labelnum, labeling, contours, GoCs = cv2.connectedComponentsWithStats(img)
        labeled_img = np.zeros((img.shape[0], img.shape[1]))
        for x, y in itertools.product(range(img.shape[0]), range(img.shape[1])):
            if contours[labeling[x, y], 4] < pix_th:
                labeling[x, y]=0
            elif contours[labeling[x, y], 4] >= pix_th and labeling[x, y] != 0:
                labeled_img[x, y]=255

        labeling = labeling.astype(np.uint8)
        labeled_img = labeled_img.astype(np.uint8)
        labeled_imgs.append(labeled_img)
    return labeled_imgs


def preprocess():
    f_size = 5
    f_sigma = 0
    for mem_fpath in membrane_img_fpath_list:
        img = cv2.imread(mem_fpath, 0)
        blured_img = cv2.GaussianBlur(img, (f_size, f_size), f_sigma)
        img_name = mem_fpath.rsplit('\\', 1)[-1]
        cv2.imwrite(os.path.join(out_dpath, 'gauss', img_name), blured_img)


def extraction_boundary():
    ret = get_binarization_parameter()
    
    thin_img_fpath_list = os.listdir('{}/thin'.format(out_dpath))
    thin_img_fpath_list = [os.path.join(out_dpath, 'thin', d) for d in sorted(thin_img_fpath_list)]

    for img_fpath in thin_img_fpath_list:
        img_name = img_fpath.rsplit('\\', 1)[-1]
        
        img = cv2.imread(img_fpath, 0)
        if img is None:
            print('not found thin image')
            continue
        _, img_th = cv2.threshold(img, ret, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(out_dpath, 'th', img_name), img_th)

        labeled_imgs = get_labeled_image(img_th, pix_ths=pix_ths)
        img_name = img_name.rsplit('.')[0]
        for i, th in enumerate(pix_ths):
            cv2.imwrite(os.path.join(out_dpath, 'label', img_name + '_th{}.tif'.format(th)), labeled_imgs[i])


def get_binarization_parameter():
    # binarization of all image to connect
    thin_img_fpath_list = os.listdir('{}/thin'.format(out_dpath))
    thin_img_fpath_list = [os.path.join(out_dpath, 'thin', d) for d in sorted(thin_img_fpath_list)]
    img_list = []
    for img_fpath in thin_img_fpath_list:
        img = cv2.imread(img_fpath, 0)
        if img is None:
            print('not found thin image')
            continue
        img_list.append(img)

    img_list = np.asarray(img_list)
    ret, img_th = cv2.threshold(img_list.flatten(), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret


def lumi_area(pix_th):
    labeled_img_fpath_list = os.listdir('{}/label'.format(out_dpath))
    labeled_img_fpath_list = [os.path.join(out_dpath, 'label', d) for d in sorted(labeled_img_fpath_list) if str(pix_th) in d]
    lumi_mat=[]
    bar_label=[]
    area_mat=[]
    for labeled_img_fpath, tj_img_fpath in zip(labeled_img_fpath_list, tj_img_fpath_list):
        labeled_img = cv2.imread(labeled_img_fpath, 0)
        tj_img = cv2.imread(tj_img_fpath, 0)
        if tj_img.shape == labeled_img.shape:
            rows = tj_img.shape[0]
            cols = tj_img.shape[1]
        else:
            print('image shape not same')
            sys.exit()

        sum_lumi=0
        sum_area=0
        for i, j in itertools.product(range(rows), range(cols)):
            if labeled_img[i, j] > 0:
                sum_lumi += tj_img[i, j]
                sum_area += 1
        lumi_mat.append(sum_lumi)
        area_mat.append(sum_area)
    
    return np.array(area_mat), np.array(lumi_mat)


def evaluation():
    result_dict = {}
    for pix_th in pix_ths:
        area_mat, lumi_mat = lumi_area(pix_th)
        result_dict['area_{}'.format(pix_th)] = area_mat
        result_dict['lumi_{}'.format(pix_th)] = lumi_mat
        result_dict['ratio_{}'.format(pix_th)] = lumi_mat / area_mat

    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv('{}/evaluation_results.csv'.format(out_dpath))


def main(flag):
    if flag == 'filter':
        preprocess()
    elif flag == 'label':
        extraction_boundary()
    elif flag == 'eval':
        evaluation()


if __name__=='__main__':
    main(args.flag)