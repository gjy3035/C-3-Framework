clc; clear all;
standard_size = [1080,1920];
in_out_size = [544,960,3];
rate_h = in_out_size(1)/standard_size(1);
rate_w = in_out_size(2)/standard_size(2);
path ='..\performed_bak\';
dst_path ='..\performed_bak\';

folder_list = dir(fullfile(path)); 
for i_folder = 3:size(folder_list,1)
    
    pngs_path =[path, '\', folder_list(i_folder).name,  '\pngs\'];
    gt_path = [path, '\', folder_list(i_folder).name,  '\mats\'];    
    den_path = [dst_path, '\', folder_list(i_folder).name,  '\csv_den_maps_k15_s4_544_960\'];  
    dst_pngs_path =[path, '\', folder_list(i_folder).name,  '\pngs_544_960\'];
    if ~exist(den_path)
        mkdir(den_path);
    end
    if ~exist(dst_pngs_path)
        mkdir(dst_pngs_path);
    end  
    img_list = dir(fullfile(pngs_path,'*.png')); 
    
    for idx = 1:size(img_list,1)
        fprintf(1,'Processing %3d: %d files\n', i_folder, idx);
        filename = img_list(idx,1).name(1:10);
        load(strcat(gt_path, filename, '.mat'));
        if isempty(image_info.location)
            im_density = zeros(in_out_size);
        else
            annPoints = [image_info.location(:,2),image_info.location(:,1)];% x,y
            annPoints(:,1) = annPoints(:,1)*rate_w;
            annPoints(:,2) = annPoints(:,2)*rate_h; 
            im_density = get_density_map_gaussian(in_out_size,annPoints,15,4.0); 
        end

        den_map = im_density(:,:,1);    
        
        input_img_name = strcat(pngs_path,filename, '.png');
%         im = imread(input_img_name);
%          im = imresize(im, [in_out_size(1), in_out_size(2)]);
%         imRGB = insertShape(im,'FilledCircle',[annPoints(:,1),annPoints(:,2),5*ones(size(annPoints(:,1)))],'Color', {'red'});
%         figure(1);imshow(imRGB);
%         figure(2);imagesc(den_map);     
            csvwrite([den_path  '/' filename '.csv'], den_map);
%         save(strcat(den_path, filename, '.mat'), 'den_map', '-v6');
%         imwrite(ind2rgb(im2uint8(mat2gray(den_map)), jet(256)),strcat(den_path, filename, 'vis.png'));
%         imwrite(im,strcat(dst_pngs_path, filename, '.png'));
        xxx=1;
    end
end
    

