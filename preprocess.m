%You should change the paths below to your convenience : 
%You can have the unprocessed data anywhere, but put the directory in
%trainpath
%When running the script do not just add the file to path,  but change the
%matlab path to the root dir of the project. Otherwise the python code
%might not work (configured to 
trainpath="/home/tom/Documents/LUT/Pattern_Recog/1st_period/digits_3d_training_data/digits_3d/training_data/";
recpath=pwd
recepdir="/preprocessed/"
allCSVs = dir(fullfile(trainpath,'*.csv'));
%Dimensional resolution of the created images
res=8

dataset=zeros(res,res,length(allCSVs));

for c=1:length(allCSVs)
    currentfile=allCSVs(c);
    f=load(fullfile(trainpath,currentfile.name));
    r=flatten(f,res);
    r=standardize(r,'b')
    dataset(:,:,c)=r;
    imwrite(r, strcat(fullfile(recpath,recepdir,currentfile.name),'.png'));
    
end


function s=standardize(img, mode)
    %Standardizes the image in one of the predefined modes :
    %'b'inary :  the intensity is >=1, new intensity is 1 at said pixel 
    %'m'inmax scaling : fetching the max and min of intensity to perform
    %'a'verage and std normalization, first b is performed and then
    %pixelwise minmax scaling
    X=size(img,1);
    Y=size(img,2);
    if mode=='b' || mode=='a'
        for x=1:X
            for y=1:Y
                if img(x,y)>=1
                    img(x,y)=1;
                end
            end
        end
        if mode=='a'
            img=(img-mean(img(:)))/std(img(:))
        end
    end
    if mode=='m'
        M=max(img(:))
        m=min(img(:))
        img=(img-m)/(M-m);
    end
    if mode=='a'
        img=(img-mean(img(:)))./std(img(:))
        
    end
    s=img;
end


function C=digit_classify(data)
    %This function predicts the class of the given 3*N matrix
    %It trains the network in python over the whole training dataset a few
    %times(until convergence or max epochs is reached
    %First we will format the test data the same way we've formatted the
    %train dataset
    r=flatten(f,res);
    r=standardize(r,'b');
    %We store it in the current working directory for the python file to
    %find it
    path_test=strcat(pwd,'test.png')
    imwrite(r, path_test);
    C=py.main.predict_digit(path_test)
end


function recep_img_2d=flatten(pos,res)
    %This function makes a 2d image from a N*3 matrix and
    %res gives out the resolution of image
    %eg. if res=8 the image will be 8*8 pixels
    
    
    %We will find max and min along each dimension to represent the digit in a 
    %3d array
    minx=min(pos(:,1));
    maxx=max(pos(:,1));
    xrange=maxx-minx;
    miny=min(pos(:,2));
    maxy=max(pos(:,2));
    yrange=maxy-miny;
    minz=min(pos(:,3));
    maxz=max(pos(:,3));
    zrange=maxz-minz;
    nb_points=size(pos,1);

    recep_size=[res res res];
    recep_img_3d=zeros(recep_size);
    voxel_pos=zeros(nb_points, 3);

    for i=1:nb_points
        proportions=[(pos(i,1)-minx)/xrange,(pos(i,2)-miny)/yrange,(pos(i,3)-minz)/zrange];
        vox=floor(proportions.*(recep_size-1))+1;
        recep_img_3d(vox(1),vox(2),vox(3))=1;
    end
    %We now have a 3d representation of the air written digit
    recep_img_2d=zeros(recep_size(1:2));

    for j=1:recep_size(3)
        recep_img_2d=recep_img_2d + recep_img_3d(:,:,j);
    end
    recep_img_2d=rot90(recep_img_2d);
end

