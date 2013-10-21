clear all;
nchannels = 8;
reduc = 4;
Reduce_factor_6_option = 1;
load rawdata_GE_brain

% if Reduce_factor_6_option == 1
%     load GE_human_brain
%    
%     %Img = Img*1000; 
%     % make 'GE_human_brain' data have the same value range as other data
%     %Img_temp = Img(3:254,:,:);
%     Img_temp=Img;
%     clear Img raw_data;
%     Img = Img_temp;
    for k = 1:nchannels
        %Img(:,:,k)=permute(Img(:,:,k),[2 1 3]);
        Img(:,:,k) = fftshift(ifft2(fftshift(raw_data(:,:,k))));
%          imcoil=abs( Img(:,:,k));
%         imcoil=imcoil./mean(imcoil(:));
        %figure,imshow(abs(imcoil), [0 5 ])
    end
%     %reduc = 6;
% end

% figure,imshow(abs(Img(:,:,1)),[])
[D1,D2,CoilNum] = size(Img);

% getting high resolution image
Image_E = zeros(D1,D2);
for k = 1:nchannels
    Image_E = Image_E + Img(:,:,k).*conj(Img(:,:,k));
end
Image_E = sqrt(Image_E);

% getting the mask
mask = zeros(D1,D2);
max_val = max(max(abs(Image_E)));
mask(find(abs(Image_E)>0.1*max_val)) = 1;
mask = imfill(mask,'holes');
mask = medfilt2(mask);
baseimg=abs(Image_E);
baseimg=baseimg./max(baseimg(:));
  figure(100),imshow(baseimg,[0 0.7])

% getting the sensitivity map using the low-resolution image
center_line_num = 32;
low_zero_padding = zeros(D1,D2,CoilNum);

ACS1=D1/2-center_line_num/2+1;
ACS2=D1/2+center_line_num/2;

%low_zero_padding(:,D1/2-center_line_num/2+1:D1/2+center_line_num/2,:)= raw_data(:,D1/2-center_line_num/2+1:D1/2+center_line_num/2,:);
low_zero_padding(ACS1:ACS2,:,:)= raw_data(ACS1:ACS2,:,:);
% using hamming window to alleviate the Gibbs effect
reduced_low=zeros(D1,D2);
reduced_both=zeros(D1,D2);
for k = 1:nchannels
        reduced_low=zeros(D1,D2);
        data_32=low_zero_padding(:,:,k);
        reduced_low(ACS1:reduc:ACS2,:)=data_32(ACS1:reduc:ACS2,:);
        reduced_both(1:reduc:end,:)=raw_data(1:reduc:end,:,k);
        recon_low(:,:,k) =fftshift(ifft2(fftshift(data_32)));
         reduc_recon_low(:,:,k) =fftshift(ifft2(fftshift(reduced_low)));
         reduc_recon_both(:,:,k) =fftshift(ifft2(fftshift(reduced_both)));
        %recon_low(:,:,k)=recon_cs_256_GE_32central(permute(data_32,[2 1 3]));
    
end
% data_mask=(abs(data_32)>0);
% figure,imshow(data_mask);
% figure,imshow(abs(reduc_recon_low(:,:,1)), [ ])
Img_Recon_final = zeros(D1,D2);
for k = 1:nchannels
    Img_Recon_final = Img_Recon_final+ recon_low(:,:,k).*conj(recon_low(:,:,k));
end
Img_Recon_final = sqrt(Img_Recon_final);
% figure;imshow(Img_Recon_final,[]);title('Final Recon Image');

 in_NN1=reshape(reduc_recon_low,D1*D2,CoilNum);
 in_NN2=cat(3,real(in_NN1),imag(in_NN1));
 in_NN3=permute(in_NN2,[3,2,1]);
 in_NN4=reshape(in_NN3,2*CoilNum,D1*D2);
    [X,Y] = meshgrid(1:D1, 1:D2);
    XY1=cat(3,Y(:),X(:));
    XY2=squeeze(permute(XY1,[3,2,1]));
    in_NN=[in_NN4;XY2];
    out_NN=(Img_Recon_final(:)).';
    tic
    net = newff(in_NN,out_NN,98);
%     ,'radbas','traincgf'
     net.trainParam.epochs = 1000;
     net.trainParam.goal = 1e-3;
      net.trainParam.lr = 1e-6;
     net.layers{1}.transferFcn = 'radbas';
 %    net = newff(minmax(in_NN),[98 1],{'tansig' 'purelin'},'traincgf');


     net = train(net,in_NN,out_NN);
    
    toc
    in_test_NN1=reshape(reduc_recon_low,D1*D2,CoilNum);
    in_test_NN2=cat(3,real(in_test_NN1),imag(in_test_NN1));
    in_test_NN3=permute(in_test_NN2,[3,2,1]);
    in_test_NN4=reshape(in_test_NN3,2*CoilNum,D1*D2);
      in_test_NN=[in_test_NN4;XY2];
    
    im_rec = sim(net,in_test_NN);
    
 
im_rec=im_rec./max(abs(im_rec(:)));
figure;imshow(reshape(im_rec,D1,D2),[0, max(im_rec(:))*0.7]);title('basic sense')
%num_line=sum(mask_cs(:,1))
%  error=error_ld(baseimg,im_rec)
%  error3=abs(baseimg-im_rec);
%  maxerror=max(error3(:))
%  figure,imshow(error3*1000,[0 255  ])
%%%%
% aliased_img_2=zeros(D1/reduc,D2,CoilNum);
% aliased_img_2=aliased_img(D1/2-D1/reduc/2+1:D1/2+D1/reduc/2,:,:);
% init_reg_img = Img_Recon_final./3;
% path(path,'StandardRegularization/regu/')
% %aliased_img_2 = permute(aliased_img_2, [2 1 3]);
% %init_reg_img = permute(Img_Recon_final,[2 1])./3; % initial image has been permuted
% %close all
% t = cputime;
% 
% %[ImgRec,CondNum,GMap,RegParameter]=SenseCartesianRegLCurve(aliased_img_2,sensitivity,reduc,1,fi_matrix,init_reg_img);
% [ImgRec,lambda,CondNum,GMap,RegParameter]=SenseCartesianRegLCurve_old(aliased_img_2,permute(sen_map,[1 2 3]),reduc,1,fi_matrix,init_reg_img); % for fixed 'lambda'
% e = cputime - t;
% ImgRec=trans180(ImgRec);
% ImgRec=abs(ImgRec)./max(abs(ImgRec(:)));
% error11=error_ld(baseimg,ImgRec)
% error33=abs(baseimg-ImgRec);
% str1=['l-curve regulized sense ' '\lambda = ' num2str(lambda)];
% figure,imshow((ImgRec),[0, 0.7]);title(str1) 
% figure,imshow(error33*1000,[0 255  ])
% maxerror11=max(error33(:))


% H=fspecial('average',3);   
% b=filter2(H,im_rec, 'same');
% figure;imshow((b),[0, 0.7]);title('median sense')
