from scipy import io as sio
import PIL.Image as Image
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter 
import scipy
import pdb



dataRoot = '/media/D/DataSet/CC/ShanghaiTech_Crowd_Detecting'

max_size=[1024,1024]

A_train = dataRoot + '/part_A_final/Train'
A_test = dataRoot + '/part_A_final/Test'

dstRoot = '/media/D/DataSet/CC/UCF_QNRF_adpt'


if not os.path.exists(dstRoot):
    os.mkdir(dstRoot)


def gaussian_filter_density(pts,dst_size):
    # print gt.shape
    density = np.zeros([dst_size[1],dst_size[0]], dtype=np.float32)
    
    if pts is None:
        return density

    gt_count = len(pts)

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros([dst_size[1],dst_size[0]], dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(dst_size))/2./2. #case: 1 point
        # pdb.set_trace()
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density




def generate_den_map(img_paths,srcRoot,dstRoot,attr):
    dstRootAttr = os.path.join(dstRoot,attr)
    if not os.path.exists(dstRootAttr):   
        os.mkdir(dstRootAttr) 
    if not os.path.exists(dstRootAttr+'/img'):   
        os.mkdir(dstRootAttr+'/img')
    if not os.path.exists(dstRootAttr+'/den'):   
        os.mkdir(dstRootAttr+'/den')


    for img_path in img_paths:
        print img_path
        mat = sio.loadmat(os.path.join(srcRoot,'ground_truth',img_path.replace('.jpg','.mat').replace('IMG_','GT_IMG_')))
        img= Image.open(os.path.join(srcRoot,'images',img_path))
        wd, ht = img.size

        dst_wd = wd/16*16
        rate_wd  = float(dst_wd)/float(wd)
        dst_ht = ht/16*16
        rate_ht = float(dst_ht)/float(ht)

        # print [ht,wd]
        # print [dst_ht,dst_wd]

        # pdb.set_trace()

        img = img.resize((dst_wd, dst_ht), Image.BILINEAR)

        img.save(os.path.join(dstRootAttr,'img',img_path))
        
        gt = mat["image_info"][0,0][0,0][0]
        

        gt_x = (gt[:,0]*rate_wd).astype(np.int64)
        gt_y = (gt[:,1]*rate_ht).astype(np.int64)

        pts = np.vstack((gt_x,gt_y)).transpose()# x,y


        filtered_pts = [(pt[0]<dst_wd and pt[1]<dst_ht) for pt in pts]

        # pdb.set_trace()
        

        pts = pts[filtered_pts,:]


        k = gaussian_filter_density(pts,[dst_wd,dst_ht])
        sio.savemat(os.path.join(dstRootAttr,'den',img_path.replace('.jpg','.mat')),{'map':k})
        


A_train_list = os.listdir(A_train+'/images')
A_train_list.sort()


A_test_list = os.listdir(A_test+'/images')
A_test_list.sort()

print 'train'


generate_den_map(A_train_list,A_train,dstRoot, 'train')

print 'test'


generate_den_map(A_test_list,A_test,dstRoot,'test')                           