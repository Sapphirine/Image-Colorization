idx = 1;
im_new = load('D:/Users/Luis/deep-learning-models/color_pic');
im_new = im_new.im_out;
im_new = squeeze(im_new);
im_old = load(sprintf('D:/Luis/Documents/mirflickr25k/train/train_x_%d.mat',idx));
im_old = im_old.im_in;
sample = imread(sprintf('D:/Luis/Documents/mirflickr25k/mirflickr/im%d.jpg',idx));

% Unbin  CIELAB color channels
edges = 50;
max_size = 224;
rgb = 3;
pic = zeros(max_size,max_size,rgb);
pic(:,:,1) = im_old;
for i = 1:max_size
    for j = 1:max_size
        pic(i,j,2) = find(im_new(i,j,:,1) == max(im_new(i,j,:,1)));
        pic(i,j,3) = find(im_new(i,j,:,2) == max(im_new(i,j,:,2)));   
    end
end
pic(:,:,2) = (pic(:,:,2)-25)*256/50;
pic(:,:,3) = (pic(:,:,3)-25)*256/50;
% Convert back to RGB
pic_rgb = lab2rgb(pic)*255;

pic_rgb = uint8(pic_rgb);
test1 = pic_rgb(:,:,1);
test2 = pic_rgb(:,:,2);
test3 = pic_rgb(:,:,3);
% Plot image
figure(1)
montage([sample(:,:,1) sample(:,:,2) sample(:,:,3)])
figure(2)
montage([pic_rgb(:,:,1) pic_rgb(:,:,2) pic_rgb(:,:,3)])
imwrite(pic_rgb,'D:/Luis/Documents/mirflickr25k/out/demo_pic.jpeg')
%}