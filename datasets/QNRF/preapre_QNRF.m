%%
clc; clear all;
maxSize = [1024,1024];

%% test set
path ='H:/UCF-QNRF_ECCV18\Test\';
output_path = '../UCF-qnrf/1024x1024_mod16/test';
train_path_img = strcat(output_path,'/', 'img/');
train_path_den = strcat(output_path,'/', 'den/');

mkdir(output_path);
mkdir(train_path_img);
mkdir(train_path_den);

for idx = 1:334
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Test Set: Processing %3d/%d files\n', idx, 334);
    end
    load(strcat(path, 'img_',num2str(i,'%04d'),'_ann.mat')) ;
    input_img_name = strcat(path,'img_',num2str(i,'%04d'),'.jpg');
    im = imread(input_img_name);  
    [h, w, c] = size(im);

    %% resize
    rate = maxSize(1)/h;
    rate_w = w*rate;
    if rate_w>maxSize(2)
        rate = maxSize(2)/w;
    end
    rate_h = double(int16(h*rate/16)*16)/h;
    rate_w = double(int16(w*rate/16)*16)/w;
    im = imresize(im,[int16(h*rate/16)*16,int16(w*rate/16)*16]);
    annPoints(:,1) = annPoints(:,1)*double(rate_w);
    annPoints(:,2) = annPoints(:,2)*double(rate_h);    
    %% generation   
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    %% visualization    
    %      imRGB = insertShape(im,'FilledCircle',[annPoints(:,1),annPoints(:,2),5*ones(size(annPoints(:,1)))],'Color', {'red'});
    %     figure(1);imshow(imRGB);
    %% save
    imwrite(im, [train_path_img num2str(idx) '.jpg']);
    csvwrite([train_path_den num2str(idx) '.csv'], im_density);
end


%% train set
path ='H:/UCF-QNRF_ECCV18/Train/';
output_path = '../UCF-qnrf-processed/train';
train_path_img = strcat(output_path,'/', 'img/');
train_path_den = strcat(output_path,'/', 'den/');

mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);

for idx = 1:1201
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Train Set: Processing %3d/%d files\n', idx, 1201);
    end
    load(strcat(path, 'img_',num2str(i,'%04d'),'_ann.mat')) ;
    input_img_name = strcat(path,'img_',num2str(i,'%04d'),'.jpg');
    im = imread(input_img_name);  
    [h, w, c] = size(im);

    %% resize
    rate = maxSize(1)/h;
    rate_w = w*rate;
    if rate_w>maxSize(2)
        rate = maxSize(2)/w;
    end
    rate_h = double(int16(h*rate/16)*16)/h;
    rate_w = double(int16(w*rate/16)*16)/w;
    im = imresize(im,[int16(h*rate/16)*16,int16(w*rate/16)*16]);
    annPoints(:,1) = annPoints(:,1)*double(rate_w);
    annPoints(:,2) = annPoints(:,2)*double(rate_h);    
    %% generation   
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    %% visualization    
    %      imRGB = insertShape(im,'FilledCircle',[annPoints(:,1),annPoints(:,2),5*ones(size(annPoints(:,1)))],'Color', {'red'});
    %     figure(1);imshow(imRGB);
    %% save
    imwrite(im, [train_path_img num2str(idx) '.jpg']);
    csvwrite([train_path_den num2str(idx) '.csv'], im_density);
end


