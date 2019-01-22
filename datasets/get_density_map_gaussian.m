function im_density = get_density_map_gaussian(im,points,k_size,sigma)


im_density = zeros(size(im)); 
[h,w] = size(im_density);

if(length(points)==0)
    return;
end
%     GJY
% if(length(points(:,1))==1)
%     x1 = max(1,min(w,round(points(1,1))));
%     y1 = max(1,min(h,round(points(1,2))));
% 
%     im_density(y1,x1) = 255;
%     return;
% end
for j = 1:size(points,1) 	
    f_sz = k_size;
%     sigma = 4.0;
    H = fspecial('Gaussian',[f_sz, f_sz],sigma);
    x = min(w,max(1,abs(int32(floor(points(j,1)))))); 
    y = min(h,max(1,abs(int32(floor(points(j,2))))));
    if(x > w || y > h)
        continue;
    end
    x1 = x - int32(floor(f_sz/2)); y1 = y - int32(floor(f_sz/2));
    x2 = x + int32(floor(f_sz/2)); y2 = y + int32(floor(f_sz/2));
    dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0;
    change_H = false;
    if(x1 < 1)
        dfx1 = abs(x1)+1;
        x1 = 1;
        change_H = true;
    end
    if(y1 < 1)
        dfy1 = abs(y1)+1;
        y1 = 1;
        change_H = true;
    end
    if(x2 > w)
        dfx2 = x2 - w;
        x2 = w;
        change_H = true;
    end
    if(y2 > h)
        dfy2 = y2 - h;
        y2 = h;
        change_H = true;
    end
    x1h = 1+dfx1; y1h = 1+dfy1; x2h = f_sz - dfx2; y2h = f_sz - dfy2;
    if (change_H == true)
        H =  fspecial('Gaussian',[double(y2h-y1h+1), double(x2h-x1h+1)],sigma);
    end
    im_density(y1:y2,x1:x2) = im_density(y1:y2,x1:x2) +  H;
     
end

end