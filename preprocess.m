trainpath="/home/tom/Documents/LUT/Pattern_Recog/digits_3d_training_data/digits_3d/training_data/";
recepdir="/preprocessed/"
allCSVs = dir(fullfile(trainpath,'*.csv'));
for c=1:length(allCSVs)
    c
    currentfile=allCSVs(c);
    f=load(fullfile(trainpath,currentfile.name));
    r=flatten(f);
    imwrite(r, strcat(fullfile(trainpath,recepdir,currentfile.name),'.png'));
    
end

function recep_img_2d=flatten(pos)
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
    res=10;
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

