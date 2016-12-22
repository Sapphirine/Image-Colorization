k = 25000;
offset = 0;
for idx = 1+offset:k+offset 
    % Load image 
    %idx = 2;
    im = imread(sprintf('D:/Luis/Documents/mirflickr25k/mirflickr/im%d.jpg',idx));
    [y,x,rgb] = size(im);
    max_size = 224;
    % Determine which way to crop
    if x<y
        small_dim = x;
    end
    if x>=y
        small_dim = y;
    end
    % Crop
    % Find center of image
    c = [y/2,x/2];
    im = im( floor(c(1)-small_dim/2+1):floor(c(1)+small_dim/2) , floor(c(2)-small_dim/2+1):floor(c(2)+small_dim/2) ,:);
    % Scale to 224x224
    im = imresize(im,[max_size max_size]);
    %figure(3)
    %imshow(im)
    % Color transform: 1=YUV; 2=CIELAB; 3=HSV
    ct = 2;
    edges = 50;
    im_out = zeros(max_size,max_size,edges,rgb-1);
    im_out_t = zeros(max_size,max_size,rgb-1);
    im_test = zeros(max_size,max_size,rgb-1);
    im_in = zeros(max_size,max_size,rgb-2);
    switch ct
        case 1
            %title('Image #7888 from MIR Flickr 250k Dataset')
            r = double(im(:,:,1));
            g = double(im(:,:,2));
            b = double(im(:,:,3));
            wr = .299;
            wg = .587;
            wb = .114;
            yp = wr*r + wg*g + wb*b;
            cr = .713*(r-yp);
            cb = .564*(b-yp);
            c1 = yp;
            c2 = cr;
            c3 = cb;
        case 2
            out = rgb2lab(im);
            c1 = out(:,:,1);
            c2 = out(:,:,2);
            c3 = out(:,:,3);
        case 3
            [c1, c2, c3] = rgb2hsv(im);
    end    
    %{
    figure(1)
    montage([im(:,:,1) im(:,:,2) im(:,:,3)])
    title('Image #7888 in RGB color space')
    figure(2)
    montage([c1 c2+128 c3+128], [0 255])
    title('Image #7888 in YUV color space')
    %}    
    % For input, CNN expects zero-centered RGB channels: subtract mean from intensity channel
    % For output, CNN compares 50 binned values for the 2 color channels 
    prep = 1;
    if prep
        im_out_t(:,:,1) = floor((c2*50/256)+25);
        im_out_t(:,:,2) = floor((c3*50/256)+25);
        % Binning the two color channels
        for i = 1:max_size
            for j = 1:max_size
                loc = im_out_t(i,j,1);
                im_out(i,j,loc,1) = 1;
                loc = im_out_t(i,j,2);
                im_out(i,j,loc,2) = 1;
            end
        end
        im_in = c1; 
    end
    % Save files
    save(sprintf('D:/Luis/Documents/mirflickr25k/train/train_x_%d.mat',idx),'im_in');
    save(sprintf('D:/Luis/Documents/mirflickr25k/train/train_y_%d.mat',idx),'im_out');
end
