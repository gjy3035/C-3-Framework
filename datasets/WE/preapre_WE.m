clc; clear all;
standard_size = [576,720];

RootPath = 'H:/CC_Matlab/WE';
DstPath = '../WE_blurred';
%% Test set

testset={'104207','200608','200702','202201','500717'};

for i_subset=1:5
    path =[RootPath '/test_frame/' testset{1,i_subset}  '\'];
    label_path =[RootPath '/test_label/' testset{1,i_subset} '\'];
    output_path =[DstPath '/test/' testset{1,i_subset}];
    train_path_img = strcat(output_path,'/', 'img/');
    train_path_den = strcat(output_path,'/', 'den/');

    mkdir(output_path)
    mkdir(train_path_img);
    mkdir(train_path_den);

    img_list = dir(fullfile(path,'*.jpg')); 
    
    load([label_path '\roi.mat']);
    x_p = maskVerticesXCoordinates;
    y_p = maskVerticesYCoordinates;

    for idx = 1:size(img_list,1)
        filename = img_list(idx,1).name;
        filename_no_ext = regexp(filename, '.jpg', 'split');
        filename_no_ext = filename_no_ext{1,1};
        folderName = filename(1:6);

        i = idx;
        if (mod(idx,10)==0)
            fprintf(1,'Test:Processing %3d/%d files\n', idx, size(img_list,1));
        end
        load(strcat(label_path, filename_no_ext, '.mat'));
        input_img_name = strcat(path,filename);
        im = imread(input_img_name);  
        [h, w, c] = size(im);
        if isempty(point_position)
            continue;
        end
        ori_annPoints = point_position;

        %% remove valid key points
        annPoints = [];
        i_real_p=0;
        for i_annP=1:size(ori_annPoints,1)
            in = inpolygon(ori_annPoints(i_annP,1),ori_annPoints(i_annP,2),x_p,y_p);
            if in==1
                i_real_p = i_real_p+1;
                annPoints(i_real_p,:)=ori_annPoints(i_annP,:);
            end
        end  
        %% ROI mask
        BW = roipoly(im,x_p,y_p);       
        back_img = im;
        back_img(:,:,1) = back_img(:,:,1).*uint8(~BW);
        back_img(:,:,2) = back_img(:,:,2).*uint8(~BW);
        back_img(:,:,3) = back_img(:,:,3).*uint8(~BW); 
        %% blur the region out of interest region
        H = fspecial('disk',5);
        back_img = imfilter(back_img,H,'replicate');
        back_img(:,:,1) = back_img(:,:,1).*uint8(~BW);
        back_img(:,:,2) = back_img(:,:,2).*uint8(~BW);
        back_img(:,:,3) = back_img(:,:,3).*uint8(~BW);  
        %% keep ROI 
        im(:,:,1) = im(:,:,1).*uint8(BW);
        im(:,:,2) = im(:,:,2).*uint8(BW);
        im(:,:,3) = im(:,:,3).*uint8(BW);
        %% restore img
        final_img = back_img+im;
        %% generation  
        im_density = get_density_map_gaussian(im,annPoints,15,4); 
        im_density = im_density(:,:,1);
        if isempty(annPoints)
            continue;
        end
        %% save
        imwrite(final_img, [train_path_img '/' filename_no_ext '.jpg']);
        csvwrite([train_path_den  '/' filename_no_ext '.csv'], im_density);
    end
     save([DstPath '/test/', testset{1,i_subset},'_roi.mat'],'BW');
end


%% Train set
path =[RootPath '/train_frame/'];
label_path =[RootPath '/train_label/'];
output_path = [DstPath '/train'];
train_path_img = strcat(output_path,'/', 'img/');
train_path_den = strcat(output_path,'/', 'den/');
mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);

img_list = dir(fullfile(path,'*.jpg')); 

for idx = 1:size(img_list,1)
    filename = img_list(idx,1).name;
    filename_no_ext = regexp(filename, '.jpg', 'split');
    filename_no_ext = filename_no_ext{1,1};
    folderName = filename(1:6);
    
    load([label_path folderName '\roi.mat']);
    x_p = maskVerticesXCoordinates;
    y_p = maskVerticesYCoordinates;
    
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Train: Processing %3d/%d files\n', idx, size(img_list,1));
    end
    load(strcat(label_path, folderName,'/', filename_no_ext, '.mat'));
    input_img_name = strcat(path,filename);
    im = imread(input_img_name);  
    [h, w, c] = size(im);
    if isempty(point_position)
        continue;
    end
    ori_annPoints = point_position;
    
    %% remove valid key points
    annPoints = [];
    i_real_p=0;
    for i_annP=1:size(ori_annPoints,1)
        in = inpolygon(ori_annPoints(i_annP,1),ori_annPoints(i_annP,2),x_p,y_p);
        if in==1
            i_real_p = i_real_p+1;
            annPoints(i_real_p,:)=ori_annPoints(i_annP,:);
        end
    end  
    %% ROI mask
    BW = roipoly(im,x_p,y_p);       
    back_img = im;
    back_img(:,:,1) = back_img(:,:,1).*uint8(~BW);
    back_img(:,:,2) = back_img(:,:,2).*uint8(~BW);
    back_img(:,:,3) = back_img(:,:,3).*uint8(~BW); 
    %% blur the region out of interest region
    H = fspecial('disk',5);
    back_img = imfilter(back_img,H,'replicate');
    back_img(:,:,1) = back_img(:,:,1).*uint8(~BW);
    back_img(:,:,2) = back_img(:,:,2).*uint8(~BW);
    back_img(:,:,3) = back_img(:,:,3).*uint8(~BW);  
    %% keep ROI 
    im(:,:,1) = im(:,:,1).*uint8(BW);
    im(:,:,2) = im(:,:,2).*uint8(BW);
    im(:,:,3) = im(:,:,3).*uint8(BW);
    %% restore img
    final_img = back_img+im;
    %% generation  
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    if isempty(annPoints)
        continue;
    end
    %% save
    imwrite(final_img, [train_path_img '/' filename_no_ext '.jpg']);
    csvwrite([train_path_den  '/' filename_no_ext '.csv'], im_density);
end
