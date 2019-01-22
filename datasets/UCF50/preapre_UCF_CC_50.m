clc; clear all;

path ='H:\CC_Matlab\UCFCrowdCountingDataset_CVPR13\UCF_CC_50\';
output_path = '../UCF/UCF_CC_50/';
train_path_img = strcat(output_path,'/', 'img/');
train_path_den = strcat(output_path,'/', 'den/');
train_path_seg = strcat(output_path,'/', 'seg/');

mkdir(output_path)
for i_folder = 1:5
    mkdir([train_path_img, num2str(i_folder)]);
    mkdir([train_path_den, num2str(i_folder)]);
    mkdir([train_path_seg, num2str(i_folder)]);
end
cnt = zeros(5,1);
for idx = 1:50
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, 50);
    end
    load(strcat(path, num2str(i),'_ann.mat')) ;
    input_img_name = strcat(path,num2str(i),'.jpg');
    im = imread(input_img_name);  
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end

    rate_h = double(int16(h/16)*16)/h;
    rate_w = double(int16(w/16)*16)/w;
    im = imresize(im,[int16(h/16)*16,int16(w/16)*16]);
    annPoints(:,1) = annPoints(:,1)*double(rate_w);
    annPoints(:,2) = annPoints(:,2)*double(rate_h);
    
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    
%     imRGB = insertShape(im,'FilledCircle',[annPoints(:,1),annPoints(:,2),5*ones(size(annPoints(:,1)))],'Color', {'red'});

   i_corss = ceil(rand(1)*5);
    while cnt(i_corss)>=10
        i_corss = ceil(rand(1)*5);
    end

    cnt(i_corss) = cnt(i_corss) +1;
    
    imwrite(im, [train_path_img num2str(i_corss) '/' num2str(idx) '.jpg']);
    csvwrite([train_path_den num2str(i_corss) '/' num2str(idx) '.csv'], im_density);
end

